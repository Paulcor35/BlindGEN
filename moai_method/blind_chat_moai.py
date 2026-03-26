import torch
import numpy as np
import os
import sys
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import yaml

# Ajout du dossier local au path pour trouver moai_seal_backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moai_method.moai_paper_implementation import MOAIPaperCKKS, inject_moai_engine

# --- CONFIGURATION GLOBALE ---
# On réutilise MAX_TOKENS du projet si possible, sinon 100
try:
    from compact_method.blind_chat_cpp import MAX_TOKENS
except ImportError:
    MAX_TOKENS = 100

FHE_SLICE_SIZE = 256 # Taille du fragment de vecteur pour la démo MOAI

class BlindChatMoai:
    def __init__(self, tokenizer, model):
        # On utilise le tokenizer et le modèle partagés
        self.tokenizer = tokenizer
        self.model = model
        
        # Initialisation du moteur MOAI (Backend C++ SEAL)
        print("  [MOAI-SERVER] Initialisation du moteur FHE MOAI (Natif)...")
        self.fhe_engine = MOAIPaperCKKS()
        
        # Injection du moteur MOAI (Activations polynomiales & LayerNorm rotation-free) sur le modèle partagé
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moai_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f: 
                config = yaml.safe_load(f)
            # Attention: cela modifie le modèle partagé pour inclure les approximations MOAI
            self.model = inject_moai_engine(self.model, "gpt2", config)
        
        self.model.eval()
        print("  [MOAI] Moteur MOAI prêt sur modèle partagé.")

    def _find_mlp_layer(self):
        # On cherche une couche MLP pour le benchmark FHE
        for name, module in self.model.named_modules():
            if "c_fc" in name or "mlp.c_fc" in name:
                return module
        return None

    def chat_stream(self, prompt, max_tokens=100, temperature=0.7, top_k=50, repetition_penalty=1.2):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
        
        target_layer = self._find_mlp_layer()
        if target_layer:
            W_fhe = target_layer.weight.data.float().cpu().numpy()[:FHE_SLICE_SIZE, :FHE_SLICE_SIZE].astype(np.float64)
            if target_layer.bias is not None:
                B_fhe = target_layer.bias.data.float().cpu().numpy()[:FHE_SLICE_SIZE].astype(np.float64)
            else:
                B_fhe = np.zeros(FHE_SLICE_SIZE)
        
        for i in range(max_tokens):
            t_token_start = time.time()
            with torch.no_grad():
                # On récupère les hidden states pour le benchmark FHE
                outputs = self.model(input_ids, output_hidden_states=True)
                
                total_fhe_time = 0.0
                last_enc_bytes_len = 0
                
                # --- ÉTAPE 2 : LE SERVEUR TRAITE LES 12 BLOCS GPT-2 EN FHE ---
                if target_layer:
                    # Dans MOAI, on traite idéalement chaque bloc Transformer entier (Attn+MLP)
                    # Ici on simule la latence cumulée des 12 opérations MOAI réelles
                    for layer_idx in range(12):
                        t_fhe_s = time.time()
                        
                        # Accès aux données de la couche (approximation pour le benchmark)
                        h_state = outputs.hidden_states[layer_idx+1][0, -1, :FHE_SLICE_SIZE].float().cpu().numpy().reshape(1, -1).astype(np.float64)
                        
                        # 1. Chiffrement packing MOAI
                        enc_h = self.fhe_engine.col_pack_encrypt(h_state)
                        last_enc_bytes_len = len(enc_h) if isinstance(enc_h, bytes) else len(str(enc_h))
                        
                        # 2. CPMM (Calcul Homomorphe MOAI)
                        _ = self.fhe_engine.he_cpmm(enc_h, W_fhe, B_fhe, self.fhe_engine.public_ctx, server=self.fhe_engine.server)
                        
                        total_fhe_time += (time.time() - t_fhe_s)
                    
                # Sampling (Greedy / Multinomial amélioré)
                logits = outputs.logits[:, -1, :].clone()
                
                # 1. Repetition Penalty
                for token_id in set(input_ids[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

                # 2. Temperature
                logits = logits / max(temperature, 1e-5)
                
                # 3. Top-K
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

                # 4. Probabilities & Multinomial
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                word = self.tokenizer.decode(next_token[0])
                
                yield {
                    "word": word,
                    "exec_time": total_fhe_time,
                    "enc_bytes_len": last_enc_bytes_len
                }
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

if __name__ == "__main__":
    # Test rapide
    moai = BlindChatMoai()
    for res in moai.chat_stream("Hello MOAI", max_tokens=10):
        print(res["word"], end="", flush=True)
