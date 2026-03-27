import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import moai_seal_backend
    HAS_MOAI_BACKEND = True
except ImportError:
    HAS_MOAI_BACKEND = False
    print("[WARN] moai_seal_backend non disponible (C++ extension non compilée)")

# SEULEMENT si on n'a pas moai_seal_backend on importe TenSEAL pour le fallback
# car importer deux versions de SEAL peut causer des conflits fatals (AbortHandler.h)
HAS_TENSEAL = False
ts = None
if not HAS_MOAI_BACKEND:
    try:
        import tenseal as ts
        HAS_TENSEAL = True
    except ImportError:
        print("[WARN] ni tenseal ni moai_seal_backend ne sont disponibles")

# ─────────────────────────────────────────────
# 1. Activation MOAI en clair (simulations)
# ─────────────────────────────────────────────
def moai_softmax_rotation_free(scores: torch.Tensor) -> torch.Tensor:
    """Simulation clair de l'Algorithm 1"""
    x_exp = torch.exp(scores - scores.detach().amax(dim=-1, keepdim=True))
    x_sum = x_exp.sum(dim=-1, keepdim=True)
    return x_exp / x_sum

class MOAILayerNorm(nn.Module):
    """Simulation clair de l'Algorithm 8"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        d = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
        self.d = d
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias   = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.d
        S = x.sum(dim=-1, keepdim=True)
        diff = d * x - S
        var_scaled = (diff * diff).sum(dim=-1, keepdim=True) / (d * d)
        inv_std = torch.rsqrt(var_scaled + self.eps)
        return self.weight * (diff * inv_std / math.sqrt(d)) + self.bias

class MOAIRMSNorm(nn.Module):
    """Adaptation MOAI Rotation-Free pour RMSNorm (utilisé par Qwen/Llama)"""
    def __init__(self, dim, eps=1e-6, use_plus_one=False):
        super().__init__()
        self.eps = eps
        self.use_plus_one = use_plus_one
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, *args, **kwargs):
        # RMSNorm = x / sqrt(mean(x^2) + eps) * weight
        d = x.shape[-1]
        ms = (x.float() * x.float()).sum(dim=-1, keepdim=True) / d
        inv_std = torch.rsqrt(ms + self.eps)
        output = x.float() * inv_std
        
        # Qwen 3.5 / Gemma trick: weight scaling (1 + weight)
        if self.use_plus_one:
            return (output * (1.0 + self.weight.float())).type_as(x)
        else:
            return (output * self.weight.float()).type_as(x)


class MoaiPolyActivation(nn.Module):
    def __init__(self, mode="GELU", a_bound=-10.0, b_bound=10.0, degree=23):
        super().__init__()
        self.mode = mode
        self.a_bound, self.b_bound = a_bound, b_bound
        if mode == "GELU":
            self.coeffs = MOAIPaperCKKS.compute_gelu_minimax_poly_coeffs(degree=degree, a=a_bound, b=b_bound)
        else:
            self.coeffs = MOAIPaperCKKS.compute_sigmoid_poly_coeffs(degree=degree, a=a_bound, b=b_bound)
        self.coeffs_horner = list(reversed(self.coeffs))

    def forward(self, h):
        x = 0.7978845608 * (h + 0.044715 * (h ** 3)) if self.mode == "GELU" else h
        x_clamped = torch.clamp(x, self.a_bound, self.b_bound)
        p_res = torch.full_like(x_clamped, self.coeffs_horner[0])
        for i in range(1, len(self.coeffs_horner)):
            p_res = p_res * x_clamped + self.coeffs_horner[i]
        return 0.5 * h * (1.0 + p_res) if self.mode == "GELU" else h * p_res

def inject_moai_engine(model, model_id, config):
    bounds = config.get("models", {}).get(model_id, config.get("default", {}))
    a_b, b_b = bounds.get("a_bound", -10.0), bounds.get("b_bound", 10.0)
    print(f"[MOAI] Simulation FHE : Intervalle [{a_b}, {b_b}]")

    for name, module in list(model.named_modules()):
        mod_name = module.__class__.__name__.upper()
        mode = "GELU" if "GELU" in mod_name else "SILU" if "SILU" in mod_name else None
        if mode:
            parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
            setattr(parent, name.rsplit('.', 1)[-1], MoaiPolyActivation(mode, a_b, b_b))
        if isinstance(module, nn.LayerNorm):
            moai_ln = MOAILayerNorm(module.normalized_shape, eps=module.eps)
            moai_ln.weight.data, moai_ln.bias.data = module.weight.data.clone(), module.bias.data.clone()
            parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
            setattr(parent, name.rsplit('.', 1)[-1], moai_ln)
    return model

class MOAIPaperCKKS:
    """
    Implémentation fidèle du pipeline FHE de l'architecture MOAI
    telle que décrite dans Zhang et al. (2025).

    Mode exclusif: 'paper' (paramètres sécurité 128-bit, 100% pure CKKS).
    """

    # Paramètres ajustés pour la compatibilité backend SEAL C++
    # Paramètres de PRODUCTION (Production Mode) - Alignés sur Compact
    _PAPER_POLY_MOD   = 16384                          # N = 2^14 (Haute Sécurité / Profondeur)
    _PAPER_COEFF_BITS = [60, 40, 40, 40, 60]           # ~240 bits (128-bit security@16384)
    _PAPER_SCALE      = 2 ** 40                         # Scale industrielle: 2^40

    def __init__(self, poly_n=16384, scale_bits=40):
        self.poly_n = poly_n
        self.scale = 2 ** scale_bits
        
        # Adaptation dynamique du modulus chain pour la sécurité 128-bit
        if poly_n <= 8192:
            self.coeff_bits = [60, 40, 60]
        elif poly_n == 16384:
            self.coeff_bits = [60, 40, 40, 40, 60]
        else: # 32768
            self.coeff_bits = [60, 40, 40, 40, 40, 40, 40, 60]

        if not HAS_MOAI_BACKEND:
            if HAS_TENSEAL:
                print(f"[INFO] Fallback TenSEAL (N={poly_n}).")
                self._init_tenseal()
            else:
                raise RuntimeError("Backend MOAI non dispo.")
        else:
            print(f"Initialisation MOAI SEAL (N={poly_n}, Scale=2^{scale_bits})...")
            # PASSAGE CRITIQUE : on transmet les bits d'échelle au C++
            self.client = moai_seal_backend.MoaiClient(self.poly_n, scale_bits)
            self.public_ctx = self.client.get_params()
            
            self.fhe_slice_size = 256 
            self.batch_size = 1
            
            self.server = moai_seal_backend.MoaiServer(self.public_ctx)
            if hasattr(self.server, "set_scale"):
                self.server.set_scale(self.scale)
                
            self.server.set_batch_size(self.batch_size)
            self.galois_keys = self.client.get_galois(self.fhe_slice_size, self.batch_size)
            self.server.set_galois(self.galois_keys)
            print(f"[OK] Moteur MOAI prêt (N={poly_n}).")

    def update_galois(self, fhe_slice_size, batch_size=1):
        """Met à jour dynamiquement les clés de Galois pour une nouvelle dimension."""
        if HAS_MOAI_BACKEND and hasattr(self, 'client'):
            self.fhe_slice_size = fhe_slice_size
            self.batch_size = batch_size
            # Régénération des clés de Galois (Rotation steps dépendent de sqrt(N))
            self.galois_keys = self.client.get_galois(self.fhe_slice_size, self.batch_size)
            self.server.set_galois(self.galois_keys)
            # On force le reset du cache de poids du serveur car la dimension a changé
            if hasattr(self.server, "__weights_init_done"):
                delattr(self.server, "__weights_init_done")
            
            # On nettoie aussi le cache de classe si présent
            if hasattr(MOAIPaperCKKS, "_server_cache") and id(self.server) in MOAIPaperCKKS._server_cache:
                MOAIPaperCKKS._server_cache.remove(id(self.server))
                
            print(f"[MOAI] Clés de Galois et Cache Poids mis à jour (N={fhe_slice_size})")

    def _init_tenseal(self):
        poly_mod = self.poly_n
        coeff_bits = self.coeff_bits
        scale = self.scale
        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod,
            coeff_mod_bit_sizes=coeff_bits,
        )
        self.ctx.generate_galois_keys()
        self.ctx.generate_relin_keys()
        self.ctx.global_scale = scale

    def public_context_bytes(self) -> bytes:
        if HAS_MOAI_BACKEND and hasattr(self, 'public_ctx'):
            return self.public_ctx
        return self.ctx.serialize(save_secret_key=False)

    # ── Packing Utilities (Definitions 3.1 & 3.2 & C.1) ─────────────
    def col_pack_encrypt(self, matrix_np):
        """Définition 3.1: Encode X en ciphertexts. Supporte le batching."""
        if HAS_MOAI_BACKEND and hasattr(self, 'client'):
            # Si matrix_np est une liste (ou matrix_np.ndim > 2), on traite comme un batch
            if isinstance(matrix_np, list):
                if len(matrix_np) != self.batch_size:
                    # On ajuste le serveur dynamiquement si besoin
                    self.batch_size = len(matrix_np)
                    self.server.set_batch_size(self.batch_size)
                    self.galois_keys = self.client.get_galois(self.fhe_slice_size, self.batch_size)
                    self.server.set_galois(self.galois_keys)
                return self.client.encrypt_batch_interleaved(matrix_np)
                
            # Mode standard (1 seul vecteur)
            if matrix_np.shape[0] == 1:
                # On force le batch_size à 1 pour être sûr
                if self.batch_size != 1:
                    self.batch_size = 1
                    self.server.set_batch_size(1)
                    self.galois_keys = self.client.get_galois(self.fhe_slice_size, 1)
                    self.server.set_galois(self.galois_keys)
                return self.client.encrypt_batch_interleaved([matrix_np.astype(np.float64).flatten()])
            
            return self.client.encrypt_columns_blob(matrix_np.astype(np.float64))
        
        m, d = matrix_np.shape
        col_ciphertexts = []
        for j in range(d):
            col_vec = matrix_np[:, j].astype(np.float64).tolist()
            enc_col = ts.ckks_vector(self.ctx, col_vec)
            col_ciphertexts.append(enc_col.serialize())
        return col_ciphertexts

    def decrypt_col_pack(self, col_data, m, d=1):
        """Déchiffre et reconstruit. Supporte le batching."""
        if HAS_MOAI_BACKEND and hasattr(self, 'client'):
            if isinstance(col_data, bytes):
                res_batch = self.client.decrypt_batch(col_data, self.batch_size, d)
                if self.batch_size == 1: return res_batch[0].reshape(1, -1)
                return res_batch
            return self.client.decrypt_batch(col_data, self.batch_size, d)
        
        # Fallback TenSEAL
        d_size = len(col_data) if d is None else d
        matrix = np.zeros((m, d_size))
        for j in range(d_size):
            vec = ts.ckks_vector_from(self.ctx, col_data[j])
            matrix[:, j] = vec.decrypt()
        return matrix

    def diag_pack_encrypt(self, matrix_np):
        """Définition 3.2: Encode X ∈ R^{m×m} en m ciphertexts."""
        m = matrix_np.shape[0]
        diag_ciphertexts = []
        for j in range(m):
            diag_vec = np.array([matrix_np[k, (k + j) % m] for k in range(m)])
            enc_diag = ts.ckks_vector(self.ctx, diag_vec.astype(np.float64).tolist())
            diag_ciphertexts.append(enc_diag.serialize())
        return diag_ciphertexts

    @staticmethod
    def interleave_batch(vectors):
        """Section 3.2 (Eq. 1): Pack N/(2m) vecteurs de taille m en un seul."""
        R = len(vectors)
        m = len(vectors[0])
        result = np.zeros(R * m, dtype=np.float64)
        for i in range(m):
            for r in range(R):
                result[i * R + r] = vectors[r][i]
        return result

    # ── Polyvals (Appendix E.2) ──────────────────────────────────────
    @staticmethod
    def compute_gelu_minimax_poly_coeffs(degree=23, a=-21, b=21):
        """
        Calcul ultra-stable via Chebyshev symétrique.
        On fit sur [-b, b] et on force les coefficients pairs à zéro.
        """
        from numpy.polynomial import Chebyshev
        
        # On définit un intervalle symétrique
        # b_bound = max(abs(a), abs(b))
        x = np.linspace(-b, b, 5000)
        y = np.tanh(x)
        
        # Fit Chebyshev (très stable)
        cheb_fit = Chebyshev.fit(x, y, deg=degree, domain=[-b, b])
        
        # On force la symétrie (tanh est impaire) en annulant les coeffs pairs
        # Les coefficients de Chebyshev c0, c1, c2...
        cheb_coeffs = cheb_fit.coef.copy()
        for i in range(0, len(cheb_coeffs), 2):
            cheb_coeffs[i] = 0.0
        
        # On recréé l'objet Chebyshev symétrisé
        cheb_fit_sym = Chebyshev(cheb_coeffs, domain=[-b, b])
        
        # Conversion finale en polynôme standard (c0, c1, c2...)
        poly_standard = cheb_fit_sym.convert(kind=np.polynomial.Polynomial)
        return poly_standard.coef.tolist()

    @staticmethod
    def compute_sigmoid_poly_coeffs(degree=23, a=-10, b=10):
        """
        Approximation polynomiale de la Sigmoïde 1/(1+exp(-x)).
        Plus stable pour SiLU (x * sigmoid(x)) car la sigmoïde est bornée.
        """
        from numpy.polynomial import Chebyshev
        x = np.linspace(a, b, 5000)
        y = 1 / (1 + np.exp(-x))
        cheb_fit = Chebyshev.fit(x, y, deg=degree, domain=[a, b])
        return cheb_fit.convert(kind=np.polynomial.Polynomial).coef.tolist()






    # ── Serveur - Aide Opérations (Halevi-Shoup / RotHE Simulation) ──
    @staticmethod
    def _he_rotate(ct, j, m):
        """Simule RotHE via matmul permutation (TenSEAL n'expose pas galois rot)."""
        if j == 0 or j % m == 0:
            return ct
        j = j % m
        P = [[0.0] * m for _ in range(m)]
        for i in range(m):
            P[(i + j) % m][i] = 1.0
        return ct.matmul(P)

    # ── Serveur - Algorithmes Cœurs du Paper ─────────────────────────
    @staticmethod
    def he_cpmm(col_X_bytes, W, bias, public_ctx_bytes, server=None):
        """Algorithm 2 : CPMM rotation-free."""
        if HAS_MOAI_BACKEND:
            # On utilise le serveur passé en paramètre pour éviter de re-instancier
            if server is None:
                server = moai_seal_backend.MoaiServer(public_ctx_bytes)
            
            W_np = np.array(W).astype(np.float64) if not isinstance(W, np.ndarray) else W.astype(np.float64)
            bias_np = np.array(bias).astype(np.float64) if not isinstance(bias, np.ndarray) else bias.astype(np.float64)
            
            # OPTIMISATION MAJEURE : Cache BSGS
            if not hasattr(server, "__weights_init_done") and not (hasattr(MOAIPaperCKKS, "_server_cache") and id(server) in MOAIPaperCKKS._server_cache):
                server.set_weights_bsgs(W_np, bias_np)
                try: 
                    server.__weights_init_done = True 
                except AttributeError:
                    if not hasattr(MOAIPaperCKKS, "_server_cache"): MOAIPaperCKKS._server_cache = set()
                    MOAIPaperCKKS._server_cache.add(id(server))
            
            # On utilise BSGS Vector Matmul très rapide (O(sqrt(N)) rotations) si X est 1 vecteur FHE périodique
            if isinstance(col_X_bytes, bytes) and W_np.shape[0] > 1:
                # Seuil à 2MB pour 1 seul ciphertext (taille approx 1.2MB pour N=16384)
                if len(col_X_bytes) < 2000000:
                    return server.he_matmul_vector_bsgs(col_X_bytes)
                else: 
                    # BLOB fallback (plusieurs ciphertexts concaténés)
                    return server.he_matmul_blob(col_X_bytes, W_np.shape[0], W_np, bias_np)
            return server.he_matmul_blob(col_X_bytes, W_np.shape[0], W_np, bias_np)

        # Fallback TenSEAL
        server_ctx = ts.context_from(public_ctx_bytes)
        col_X = [ts.ckks_vector_from(server_ctx, cb) for cb in col_X_bytes]
        W = np.array(W) if not isinstance(W, np.ndarray) else W
        bias = bias.tolist() if isinstance(bias, np.ndarray) else bias
        d, d_prime = W.shape

        result_cols = []
        for j in range(d_prime):
            ct_j = None
            for i in range(d):
                w_ij = float(W[i, j])
                if abs(w_ij) > 1e-15:
                    scaled = col_X[i] * w_ij
                    ct_j = scaled if ct_j is None else ct_j + scaled
            if ct_j is not None:
                ct_j = ct_j + bias[j]
            result_cols.append(ct_j.serialize())
        return result_cols

    @staticmethod
    def he_ccmm_col_to_diag(col_Q, col_K, m):
        """Algorithm 3 : CCMM col→diag pour QK^T."""
        d_prime = len(col_Q)
        diag_QKT = []
        for j in range(m):
            ct_diag_j = None
            for i in range(d_prime):
                rotated_ki = col_K[i] if j == 0 else MOAIPaperCKKS._he_rotate(col_K[i], j, m)
                product = col_Q[i] * rotated_ki
                ct_diag_j = product if ct_diag_j is None else ct_diag_j + product
            diag_QKT.append(ct_diag_j)
        return diag_QKT

    @staticmethod
    def he_softmax_rotation_free(diag_ciphertexts, r_exp=4, iters_goldschmidt=10):
        """Algorithm 1 : Softmax rotation-free (Eq. Lemme 4.1)."""
        divisor = 2 ** r_exp
        exp_ciphertexts = []
        for ct in diag_ciphertexts:
            exp_ct = ct * (1.0 / divisor) + 1.0
            for _ in range(r_exp):
                exp_ct = exp_ct.square()
            exp_ciphertexts.append(exp_ct)

        ct_sum = exp_ciphertexts[0].copy()
        for i in range(1, len(exp_ciphertexts)):
            ct_sum += exp_ciphertexts[i]

        D = ct_sum * (1.0 / 1024.0)
        N = 1.0 / 1024.0
        for _ in range(iters_goldschmidt):
            F = D * (-1.0) + 2.0
            D = D * F
            N = F * N if isinstance(N, float) else N * F
            
        softmax_ciphertexts = [(exp_ct * N) for exp_ct in exp_ciphertexts]
        return softmax_ciphertexts

    @staticmethod
    def he_ccmm_diag_col_to_col(diag_C, col_V, m):
        """Algorithm 4 : softmax(QK)·V (CCMM diag+col→col, BSGS)."""
        b = math.ceil(math.sqrt(m))
        g = math.ceil(m / b)
        d_prime = len(col_V)
        col_CV = []

        for j in range(d_prime):
            baby = [col_V[j] if i == 0 else MOAIPaperCKKS._he_rotate(col_V[j], i, m) for i in range(b)]
            ct_j = None

            for alpha in range(g):
                inner_sum = None
                for r in range(b):
                    idx = alpha * b + r
                    if idx >= m: break
                    rot_amount = (m - alpha * b) % m
                    rot_c = diag_C[idx] if rot_amount == 0 else MOAIPaperCKKS._he_rotate(diag_C[idx], rot_amount, m)
                    prod = rot_c * baby[r]
                    inner_sum = prod if inner_sum is None else inner_sum + prod

                if inner_sum is None: continue
                if alpha * b != 0: inner_sum = MOAIPaperCKKS._he_rotate(inner_sum, alpha * b, m)
                ct_j = inner_sum if ct_j is None else ct_j + inner_sum

            col_CV.append(ct_j)
        return col_CV

    @staticmethod
    def he_layernorm_rotation_free(col_ciphertexts, gamma, beta, d, t=1, iters=10):
        """Algorithm 8 : LayerNorm rotation-free."""
        ct_S = col_ciphertexts[0].copy()
        for j in range(1, d):
            if j < len(col_ciphertexts): ct_S += col_ciphertexts[j]

        ct_d3var = None
        tmp_list = []
        for j in range(d):
            if j < len(col_ciphertexts):
                tmp_j = col_ciphertexts[j] * float(d) - ct_S
                tmp_list.append(tmp_j)
                tmp_j_sq = tmp_j.square()
                ct_d3var = tmp_j_sq if ct_d3var is None else ct_d3var + tmp_j_sq

        if t == 1:
            ct_scaled = ct_d3var * (1.0 / (d * d))
            final_gamma = gamma / np.sqrt(d)
        else:
            ct_scaled = ct_d3var * (1.0 / (d * d * d))
            final_gamma = gamma / d

        ct_a = ct_scaled * 1.0  # init initial_guess=1.0²
        ct_b_scalar = 1.0
        for _ in range(iters):
            ct_3_minus_a = ct_a * (-1.0) + 3.0
            ct_a = ct_a * ct_3_minus_a * 0.25
            if isinstance(ct_b_scalar, float):
                ct_b = ct_3_minus_a * (ct_b_scalar * 0.5)
                ct_b_scalar = None
            else:
                ct_b = ct_b * ct_3_minus_a * 0.5

        result_cols = [(tmp_list[j] * ct_b * final_gamma + beta) for j in range(len(tmp_list))]
        return result_cols

    # ── Serveur - Multi-Head Attention Complète ──────────────────────
    @staticmethod
    def he_multihead_attention(col_X, W_Q, b_Q, W_K, b_K, W_V, b_V,
                               W_out, b_out, ln_gamma, ln_beta,
                               m, d, H, public_ctx_bytes):
        """Intégration complète (Paper MOAI, Figure 2)."""
        d_prime = d // H
        scale = 1.0 / np.sqrt(d_prime)
        all_head_outputs = []

        for h in range(H):
            start, end = h * d_prime, (h + 1) * d_prime
            W_Qh, b_Qh = W_Q[:, start:end], b_Q[start:end]
            W_Kh, b_Kh = W_K[:, start:end] * scale, b_K[start:end] * scale
            W_Vh, b_Vh = W_V[:, start:end], b_V[start:end]

            # 1-3. CPMM
            col_Q = MOAIPaperCKKS.he_cpmm([x.serialize() for x in col_X], W_Qh, b_Qh, public_ctx_bytes)
            col_K = MOAIPaperCKKS.he_cpmm([x.serialize() for x in col_X], W_Kh, b_Kh, public_ctx_bytes)
            col_V = MOAIPaperCKKS.he_cpmm([x.serialize() for x in col_X], W_Vh, b_Vh, public_ctx_bytes)

            ctx = ts.context_from(public_ctx_bytes)
            col_Q = [ts.ckks_vector_from(ctx, xb) for xb in col_Q]
            col_K = [ts.ckks_vector_from(ctx, xb) for xb in col_K]
            col_V = [ts.ckks_vector_from(ctx, xb) for xb in col_V]

            # 4. Algo 3
            diag_QKT = MOAIPaperCKKS.he_ccmm_col_to_diag(col_Q, col_K, m)
            # 5. Algo 1
            diag_softmax = MOAIPaperCKKS.he_softmax_rotation_free(diag_QKT)
            # 6. Algo 4
            col_head_out = MOAIPaperCKKS.he_ccmm_diag_col_to_col(diag_softmax, col_V, m)
            all_head_outputs.append(col_head_out)

        # 7. Concat & Output Projection
        col_concat = [ct for h in range(H) for ct in all_head_outputs[h]]
        col_attn_out = MOAIPaperCKKS.he_cpmm([x.serialize() for x in col_concat], W_out, b_out, public_ctx_bytes)
        col_attn_out = [ts.ckks_vector_from(ts.context_from(public_ctx_bytes), cb) for cb in col_attn_out]

        # 8. Residual & LayerNorm (Algo 8)
        col_residual = [col_X[j] + col_attn_out[j] for j in range(d)]
        col_normed = MOAIPaperCKKS.he_layernorm_rotation_free(col_residual, ln_gamma, ln_beta, d, t=1)
        return col_normed

    @staticmethod
    def he_bootstrap(ct, public_ctx_bytes=None):
        """Algorithm 9 : Bootstrapping natif SEAL/Phantom (Placeholder TenSEAL)."""
        raise NotImplementedError("TenSEAL ne supporte pas le bootstrapping CKKS natif.")
