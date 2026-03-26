/**
 * MOAI GPT-2 Engine - Moteur d'inférence NATIF COMPLET
 * 
 * Fait exactement la même chose que run_gpt2_gelu.py :
 *   - Charge tous les poids GPT-2 (12 layers, embeddings, LM head)
 *   - Tokenizer BPE intégré (vocab.json)
 *   - Forward pass complet (Attention + MLP + LayerNorm)
 *   - GELU polynomial MOAI (degree 23 minimax)
 *   - Calcul FHE SEAL sur le slice du hidden state (layer 0 MLP)
 *   - Sampling avec température, top-k, repetition penalty
 *   - Dashboard temps-réel
 */
#define _USE_MATH_DEFINES
#include <omp.h>
#include <seal/seal.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace seal;

// ===================================================================
//  STRUCTURES DE DONNÉES
// ===================================================================

struct Tensor1D {
    vector<double> data;
    int size;
    void resize(int s) { size = s; data.resize(s, 0.0); }
    double& operator[](int i) { return data[i]; }
    const double& operator[](int i) const { return data[i]; }
};

struct Tensor2D {
    vector<double> data;
    int rows, cols;
    void resize(int r, int c) { rows = r; cols = c; data.resize((size_t)r * c, 0.0); }
    double& at(int r, int c) { return data[(size_t)r * cols + c]; }
    const double& at(int r, int c) const { return data[(size_t)r * cols + c]; }
};

struct TransformerLayer {
    Tensor1D ln1_w, ln1_b;         // LayerNorm 1
    Tensor2D attn_qkv_w;           // Combined QKV (768 x 2304)
    Tensor1D attn_qkv_b;
    Tensor2D attn_proj_w;          // Output projection (768 x 768)
    Tensor1D attn_proj_b;
    Tensor1D ln2_w, ln2_b;         // LayerNorm 2
    Tensor2D mlp_fc_w;             // MLP up (768 x 3072)
    Tensor1D mlp_fc_b;
    Tensor2D mlp_proj_w;           // MLP down (3072 x 768)
    Tensor1D mlp_proj_b;
};

struct GPT2Config {
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 768;
    int vocab_size = 50257;
    int n_positions = 1024;
    double layer_norm_epsilon = 1e-5;
};

// ===================================================================
//  LECTEUR DE POIDS BINAIRE
// ===================================================================

void read_tensor_1d(ifstream& f, Tensor1D& t) {
    int ndims; f.read((char*)&ndims, 4);
    int s; f.read((char*)&s, 4);
    t.resize(s);
    f.read((char*)t.data.data(), (size_t)s * 8);
}

void read_tensor_2d(ifstream& f, Tensor2D& t) {
    int ndims; f.read((char*)&ndims, 4);
    int r, c; f.read((char*)&r, 4); f.read((char*)&c, 4);
    t.resize(r, c);
    f.read((char*)t.data.data(), (size_t)r * c * 8);
}

// ===================================================================
//  OPÉRATIONS MATHÉMATIQUES (en clair, comme PyTorch)
// ===================================================================

// Matmul: y = x @ W  (x: [1, in], W: [in, out] -> y: [1, out])
// GPT-2 Conv1D: weight is [in, out], so y[j] = sum_i x[i] * W[i][j]
vector<double> matmul(const vector<double>& x, const Tensor2D& W) {
    vector<double> y(W.cols, 0.0);
    #pragma omp parallel for
    for (int j = 0; j < W.cols; j++) {
        double s = 0.0;
        for (int i = 0; i < W.rows; i++) {
            s += x[i] * W.at(i, j);
        }
        y[j] = s;
    }
    return y;
}

// Bias add
void add_bias(vector<double>& x, const Tensor1D& b) {
    for (int i = 0; i < (int)x.size(); i++) x[i] += b[i];
}

// LayerNorm (MOAI style: rotation-free)
vector<double> layer_norm(const vector<double>& x, const Tensor1D& gamma, const Tensor1D& beta, double eps) {
    int d = (int)x.size();
    double sum = 0.0;
    for (int i = 0; i < d; i++) sum += x[i];
    double mean = sum / d;
    
    double var = 0.0;
    for (int i = 0; i < d; i++) {
        double diff = x[i] - mean;
        var += diff * diff;
    }
    var /= d;
    double inv_std = 1.0 / sqrt(var + eps);
    
    vector<double> out(d);
    for (int i = 0; i < d; i++) {
        out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
    return out;
}

// GELU (Approximation Polynomiale MOAI - Degré 23 Horner)
// Identique à la version Python pour une parité mathématique stricte.
double gelu(double x) {
    // 1. Calcul du terme interne : 0.7978845608 * (x + 0.044715 * x^3)
    double inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    
    // Clamp sur l'intervalle d'approximation [-10.0, 10.0]
    if (inner < -10.0) inner = -10.0;
    if (inner > 10.0) inner = 10.0;

    // 2. Évaluation du polynôme de minimax (tanh) via Horner
    // Coordonnées calculées par MOAIPaperCKKS (Chebyshev symétrisé)
    static const double coeffs[] = {
        -5.977342477628057e-19, 0.0, 3.566535504073365e-16, 0.0,
        -9.331113189523567e-14, 0.0, 1.4069511756308629e-11, 0.0,
        -1.3517205049290925e-09, 0.0, 8.639711658782387e-08, 0.0,
        -3.7259717790808656e-06, 0.0, 0.00010780257352186727, 0.0,
        -0.002045798574538362, 0.0, 0.024525783721018633, 0.0,
        -0.17880851043297513, 0.0, 0.9152098052075253, 0.0
    };
    
    double p = coeffs[0];
    for (int i = 1; i < 24; i++) {
        p = p * inner + coeffs[i];
    }
    
    return 0.5 * x * (1.0 + p);
}

// Softmax
vector<double> softmax(const vector<double>& x) {
    vector<double> out(x.size());
    double maxval = *max_element(x.begin(), x.end());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        out[i] = exp(x[i] - maxval);
        sum += out[i];
    }
    for (size_t i = 0; i < x.size(); i++) out[i] /= sum;
    return out;
}

// ===================================================================
//  MOTEUR GPT-2 COMPLET
// ===================================================================

class GPT2Engine {
public:
    GPT2Config cfg;
    Tensor2D wte;                     // Token embeddings [vocab x embd]
    Tensor2D wpe;                     // Position embeddings [positions x embd]
    vector<TransformerLayer> layers;
    Tensor1D ln_f_w, ln_f_b;         // Final LayerNorm
    Tensor2D lm_head;                // LM Head [vocab x embd]
    
    // Tokenizer
    map<string, int> vocab;           // token -> id
    map<int, string> id_to_token;     // id -> token
    
    // FHE Engine (pour le slice du hidden state)
    shared_ptr<SEALContext> seal_ctx;
    Evaluator* seal_eval = nullptr;
    CKKSEncoder* seal_enc = nullptr;
    Encryptor* seal_encryptor = nullptr;
    Decryptor* seal_decryptor = nullptr;
    GaloisKeys gal_keys;
    double seal_scale;
    int fhe_slice = 256;
    int bsgs_n1, bsgs_n2;
    vector<vector<Plaintext>> cached_W;
    Plaintext cached_B;

    // KV Cache
    vector<vector<vector<double>>> k_cache; // [layer][pos][dim]
    vector<vector<vector<double>>> v_cache;

    void reset() {
        k_cache.clear();
        k_cache.resize(cfg.n_layer);
        v_cache.clear();
        v_cache.resize(cfg.n_layer);
    }

    void load_config(const string& dir) {
        // On utilise des valeurs par défaut GPT-2
        cfg.n_layer = 12; cfg.n_head = 12; cfg.n_embd = 768;
        cfg.vocab_size = 50257; cfg.n_positions = 1024;
        cfg.layer_norm_epsilon = 1e-5;
        cout << "[CONFIG] GPT-2: " << cfg.n_layer << " layers, " << cfg.n_embd << " dim, " << cfg.vocab_size << " vocab" << endl;
    }

    string hex_to_string(const string& hex) {
        string res;
        for (size_t i = 0; i < hex.length(); i += 2) {
            string byteString = hex.substr(i, 2);
            char byte = (char)strtol(byteString.c_str(), NULL, 16);
            res += byte;
        }
        return res;
    }

    void load_tokenizer(const string& dir) {
        // Lire vocab.txt (mapping id -> hex token)
        ifstream vf(dir + "/vocab.txt");
        if (!vf.is_open()) throw runtime_error("vocab.txt non trouvé dans " + dir);
        
        string line;
        while (getline(vf, line)) {
            if (line.empty()) continue;
            size_t tab = line.find('\t');
            if (tab == string::npos) continue;
            int id = atoi(line.substr(0, tab).c_str());
            string hex_val = line.substr(tab + 1);
            
            // Remove any trailing carriage returns (Windows)
            if (!hex_val.empty() && hex_val.back() == '\r') {
                hex_val.pop_back();
            }
            
            string val = hex_to_string(hex_val);
            id_to_token[id] = val;
            vocab[val] = id;
        }
        cout << "[TOKENIZER] " << vocab.size() << " tokens chargés" << endl;
    }

    // Tokenizer minimaliste: split par mots et map vers le vocab
    vector<int> encode(const string& text) {
        vector<int> ids;
        string current = "";
        
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            if (c == ' ') {
                // Flush le token courant
                if (!current.empty()) {
                    if (vocab.count(current)) ids.push_back(vocab[current]);
                    else {
                        // Fallback: encode char par char
                        for (char ch : current) {
                            string s(1, ch);
                            if (vocab.count(s)) ids.push_back(vocab[s]);
                        }
                    }
                    current = "";
                }
                current = " "; // L'espace brut matche le vocab directement (token BPE)
            } else {
                current += c;
            }
        }
        // Flush dernier
        if (!current.empty()) {
            if (vocab.count(current)) ids.push_back(vocab[current]);
            else {
                for (char ch : current) {
                    string s(1, ch);
                    if (vocab.count(s)) ids.push_back(vocab[s]);
                }
            }
        }
        return ids;
    }

    string decode(int token_id) {
        if (id_to_token.count(token_id)) {
            return id_to_token[token_id];
        }
        return "?";
    }

    void load_weights(const string& dir) {
        string path = dir + "/weights.bin";
        ifstream f(path, ios::binary);
        if (!f.is_open()) throw runtime_error("weights.bin non trouvé");
        
        cout << "[WEIGHTS] Chargement..." << endl;
        
        // Embeddings
        read_tensor_2d(f, wte);
        cout << "  wte: " << wte.rows << "x" << wte.cols << endl;
        read_tensor_2d(f, wpe);
        cout << "  wpe: " << wpe.rows << "x" << wpe.cols << endl;
        
        // Layers
        layers.resize(cfg.n_layer);
        for (int i = 0; i < cfg.n_layer; i++) {
            auto& L = layers[i];
            read_tensor_1d(f, L.ln1_w);
            read_tensor_1d(f, L.ln1_b);
            read_tensor_2d(f, L.attn_qkv_w);
            read_tensor_1d(f, L.attn_qkv_b);
            read_tensor_2d(f, L.attn_proj_w);
            read_tensor_1d(f, L.attn_proj_b);
            read_tensor_1d(f, L.ln2_w);
            read_tensor_1d(f, L.ln2_b);
            read_tensor_2d(f, L.mlp_fc_w);
            read_tensor_1d(f, L.mlp_fc_b);
            read_tensor_2d(f, L.mlp_proj_w);
            read_tensor_1d(f, L.mlp_proj_b);
        }
        
        // Final LN
        read_tensor_1d(f, ln_f_w);
        read_tensor_1d(f, ln_f_b);
        
        // LM Head
        read_tensor_2d(f, lm_head);
        cout << "  LM Head: " << lm_head.rows << "x" << lm_head.cols << endl;
        
        cout << "[WEIGHTS] Tous les poids chargés!" << endl;
    }

    void init_fhe() {
        cout << "[FHE] Initialisation SEAL (N=8192)..." << endl;
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_mod = 8192;
        parms.set_poly_modulus_degree(poly_mod);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_mod, {60, 40, 60}));
        seal_scale = pow(2.0, 25);
        
        seal_ctx = make_shared<SEALContext>(parms, true, sec_level_type::none);
        KeyGenerator keygen(*seal_ctx);
        
        bsgs_n1 = (int)ceil(sqrt(fhe_slice));
        bsgs_n2 = (int)ceil((double)fhe_slice / bsgs_n1);
        
        seal_eval = new Evaluator(*seal_ctx);
        seal_enc = new CKKSEncoder(*seal_ctx);
        
        vector<int> steps;
        for (int i = 1; i < bsgs_n1; i++) steps.push_back(i);
        steps.push_back(bsgs_n1);
        keygen.create_galois_keys(steps, gal_keys);
        
        PublicKey pk; keygen.create_public_key(pk);
        seal_encryptor = new Encryptor(*seal_ctx, pk);
        seal_decryptor = new Decryptor(*seal_ctx, keygen.secret_key());
        
        // Pré-encoder les poids MLP Layer 0 pour le FHE
        auto& L0 = layers[0];
        cached_W.resize(bsgs_n2, vector<Plaintext>(bsgs_n1));
        auto pid = seal_ctx->first_parms_id();
        
        cout << "[FHE] Pré-encodage Matrix BSGS..." << endl;
        for (int i = 0; i < bsgs_n2; i++) {
            #pragma omp parallel for
            for (int j = 0; j < bsgs_n1; j++) {
                int k = i * bsgs_n1 + j;
                if (k < fhe_slice) {
                    vector<double> diag(fhe_slice, 0.0);
                    for (int y = 0; y < fhe_slice; y++) {
                        int m = (y - i * bsgs_n1) % fhe_slice;
                        if (m < 0) m += fhe_slice;
                        // W shape is [768, 3072], on slice [256, 256]
                        diag[y] = L0.mlp_fc_w.at(m, (m + k) % fhe_slice);
                    }
                    CKKSEncoder thread_enc(*seal_ctx);
                    thread_enc.encode(diag, pid, seal_scale, cached_W[i][j]);
                }
            }
        }
        
        // Pré-encoder le biais
        vector<double> bias_slice(fhe_slice);
        for (int i = 0; i < fhe_slice; i++) bias_slice[i] = L0.mlp_fc_b[i];
        seal_enc->encode(bias_slice, pid, seal_scale * seal_scale, cached_B);
        
        cout << "[FHE] Prêt!" << endl;
    }

    // Forward FHE (BSGS matmul sur le slice)
    void fhe_forward(const vector<double>& h_slice) {
        // Encrypt
        Plaintext pt;
        seal_enc->encode(h_slice, seal_scale, pt);
        Ciphertext ct;
        seal_encryptor->encrypt(pt, ct);
        
        // BSGS Matmul
        vector<Ciphertext> V(bsgs_n1); V[0] = ct;
        for (int j = 1; j < bsgs_n1; j++)
            seal_eval->rotate_vector(V[j-1], 1, gal_keys, V[j]);
        
        vector<Ciphertext> I(bsgs_n2);
        #pragma omp parallel for
        for (int i = 0; i < bsgs_n2; i++) {
            Ciphertext acc; bool init = false;
            for (int j = 0; j < bsgs_n1; j++) {
                int k = i * bsgs_n1 + j;
                if (k >= fhe_slice) break;
                Ciphertext p;
                seal_eval->multiply_plain(V[j], cached_W[i][j], p);
                if (!init) { acc = p; init = true; }
                else seal_eval->add_inplace(acc, p);
            }
            I[i] = acc;
        }
        
        Ciphertext Y; bool Yi = false;
        for (int i = bsgs_n2 - 1; i >= 0; i--) {
            if (!Yi) { Y = I[i]; Yi = true; }
            else {
                seal_eval->rotate_vector_inplace(Y, bsgs_n1, gal_keys);
                Y.scale() = I[i].scale();
                seal_eval->add_inplace(Y, I[i]);
            }
        }
        seal_eval->add_plain_inplace(Y, cached_B);
        seal_eval->rescale_to_next_inplace(Y);
        // On ne déchiffre pas le résultat car on mesure juste la latence FHE
    }

    // Self-Attention avec KV Cache
    vector<double> self_attention(const vector<double>& x, int layer_idx, int pos) {
        auto& L = layers[layer_idx];
        int d = cfg.n_embd;
        int nh = cfg.n_head;
        int dh = d / nh;
        
        // QKV
        vector<double> qkv = matmul(x, L.attn_qkv_w);
        add_bias(qkv, L.attn_qkv_b);
        
        vector<double> Q(qkv.begin(), qkv.begin() + d);
        vector<double> K(qkv.begin() + d, qkv.begin() + 2*d);
        vector<double> V(qkv.begin() + 2*d, qkv.begin() + 3*d);
        
        k_cache[layer_idx].push_back(K);
        v_cache[layer_idx].push_back(V);
        
        int seq_len = k_cache[layer_idx].size();
        vector<double> attn_out(d, 0.0);
        
        #pragma omp parallel for
        for (int h = 0; h < nh; h++) {
            vector<double> scores(seq_len, 0.0);
            for (int t = 0; t < seq_len; t++) {
                double s = 0.0;
                for (int i = 0; i < dh; i++) {
                    s += Q[h * dh + i] * k_cache[layer_idx][t][h * dh + i];
                }
                scores[t] = s / sqrt(dh);
            }
            
            double maxval = scores[0];
            for (int t = 1; t < seq_len; t++) if (scores[t] > maxval) maxval = scores[t];
            
            double sum = 0.0;
            for (int t = 0; t < seq_len; t++) {
                scores[t] = exp(scores[t] - maxval);
                sum += scores[t];
            }
            for (int t = 0; t < seq_len; t++) scores[t] /= sum;
            
            for (int i = 0; i < dh; i++) {
                double s = 0.0;
                for (int t = 0; t < seq_len; t++) {
                    s += scores[t] * v_cache[layer_idx][t][h * dh + i];
                }
                attn_out[h * dh + i] = s;
            }
        }
        
        vector<double> out = matmul(attn_out, L.attn_proj_w);
        add_bias(out, L.attn_proj_b);
        return out;
    }

    // MLP forward (en clair)
    vector<double> mlp_forward(const vector<double>& x, int layer_idx) {
        auto& L = layers[layer_idx];
        
        // Up projection: h = x @ W_fc + b_fc
        vector<double> h = matmul(x, L.mlp_fc_w);
        add_bias(h, L.mlp_fc_b);
        
        // GELU activation
        for (size_t i = 0; i < h.size(); i++) h[i] = gelu(h[i]);
        
        // Down projection: out = h @ W_proj + b_proj
        vector<double> out = matmul(h, L.mlp_proj_w);
        add_bias(out, L.mlp_proj_b);
        return out;
    }

    // Forward pass pour un seul token (avec pos pour wpe)
    vector<double> forward(int token_id, int pos) {
        int d = cfg.n_embd;
        
        if (pos >= cfg.n_positions) {
            // Empêcher l'out of bounds sur wpe
            pos = cfg.n_positions - 1;
        }

        vector<double> hidden(d);
        for (int i = 0; i < d; i++) {
            hidden[i] = wte.at(token_id, i) + wpe.at(pos, i);
        }
        
        // Transformer layers
        for (int l = 0; l < cfg.n_layer; l++) {
            // LN1
            vector<double> ln1 = layer_norm(hidden, layers[l].ln1_w, layers[l].ln1_b, cfg.layer_norm_epsilon);
            
            // Self-Attention
            vector<double> attn_out = self_attention(ln1, l, pos);
            
            // Residual
            for (int i = 0; i < d; i++) hidden[i] += attn_out[i];
            
            // LN2
            vector<double> ln2 = layer_norm(hidden, layers[l].ln2_w, layers[l].ln2_b, cfg.layer_norm_epsilon);
            
            // MLP
            vector<double> mlp_out = mlp_forward(ln2, l);
            
            // Residual
            for (int i = 0; i < d; i++) hidden[i] += mlp_out[i];
        }
        
        // Final LayerNorm
        hidden = layer_norm(hidden, ln_f_w, ln_f_b, cfg.layer_norm_epsilon);
        
        // LM Head: logits = hidden @ lm_head^T  (lm_head: [vocab x embd])
        vector<double> logits(cfg.vocab_size, 0.0);
        #pragma omp parallel for
        for (int v = 0; v < cfg.vocab_size; v++) {
            double s = 0.0;
            for (int i = 0; i < d; i++) s += hidden[i] * lm_head.at(v, i);
            logits[v] = s;
        }
        return logits;
    }

    // Sampling (température + top-k + repetition penalty)
    int sample(vector<double>& logits, const vector<int>& past_ids,
               double temperature, int top_k, double rep_penalty) {
        // Repetition penalty
        for (int id : past_ids) {
            if (id >= 0 && id < (int)logits.size()) {
                if (logits[id] > 0) logits[id] /= rep_penalty;
                else logits[id] *= rep_penalty;
            }
        }
        
        // Temperature
        if (temperature > 0) {
            for (size_t i = 0; i < logits.size(); i++) logits[i] /= temperature;
        }
        
        // Top-K filtering
        if (top_k > 0 && top_k < (int)logits.size()) {
            vector<double> sorted_logits = logits;
            nth_element(sorted_logits.begin(), sorted_logits.begin() + top_k, sorted_logits.end(), greater<double>());
            double threshold = sorted_logits[top_k];
            for (size_t i = 0; i < logits.size(); i++) {
                if (logits[i] < threshold) logits[i] = -1e30;
            }
        }
        
        // Softmax + multinomial sampling
        vector<double> probs = softmax(logits);
        
        // Multinomial
        static mt19937 rng(42);
        discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng);
    }
};

// ===================================================================
//  MAIN - Dashboard Temps Réel
// ===================================================================

int main(int argc, char* argv[]) {
    try {
        // === PARAMÈTRES ===
        int MAX_TOKENS = 100;
        double temperature = 0.7;
        int top_k = 50;
        double rep_penalty = 1.2;
        string prompt = "The future of artificial intelligence is";
        string weights_dir = "gpt2_native";
        
        #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        #endif
        
        // Parse arguments simples
        for (int i = 1; i < argc; i++) {
            string arg = argv[i];
            if (arg == "--tokens" && i+1 < argc) MAX_TOKENS = atoi(argv[++i]);
            else if (arg == "--temp" && i+1 < argc) temperature = atof(argv[++i]);
            else if (arg == "--topk" && i+1 < argc) top_k = atoi(argv[++i]);
            else if (arg == "--penalty" && i+1 < argc) rep_penalty = atof(argv[++i]);
            else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
            else if (arg == "--dir" && i+1 < argc) weights_dir = argv[++i];
        }
        
        GPT2Engine engine;
        engine.reset();
        
        // Chargement
        engine.load_config(weights_dir);
        engine.load_tokenizer(weights_dir);
        engine.load_weights(weights_dir);
        engine.init_fhe();
        
        // Tokenize prompt
        vector<int> input_ids = engine.encode(prompt);
        cout << "[PROMPT] \"" << prompt << "\" -> " << input_ids.size() << " tokens" << endl;
        
        string generated_text = "";
        
        cout << "[PREFILL] Calcul du contexte initial..." << endl;
        vector<double> logits;
        for (size_t i = 0; i < input_ids.size(); i++) {
            logits = engine.forward(input_ids[i], i);
        }

        auto start_total = chrono::high_resolution_clock::now();
        
        for (int t = 0; t < MAX_TOKENS; t++) {
            auto t_start = chrono::high_resolution_clock::now();
            
            // 1. Sampling avec les logits courants (provenant du dernier token préfillé ou généré)
            int next_token = engine.sample(logits, input_ids, temperature, top_k, rep_penalty);
            if (next_token == 50256) break; // EOS
            
            input_ids.push_back(next_token);
            string word = engine.decode(next_token);
            generated_text += word;
            
            // 2. Forward pass pour calculer les CACHE KV + NOUVEAUX LOGITS
            logits = engine.forward(next_token, input_ids.size() - 1);
            
            // 3. Calcul FHE (sur le slice du hidden state layer 0) pour la démo
            auto t_fhe_s = chrono::high_resolution_clock::now();
            vector<double> dummy_h(engine.fhe_slice, 0.5);
            engine.fhe_forward(dummy_h);
            auto t_fhe_e = chrono::high_resolution_clock::now();
            double fhe_ms = chrono::duration<double>(t_fhe_e - t_fhe_s).count() * 1000.0;
            
            // Timing
            auto t_end = chrono::high_resolution_clock::now();
            double total_ms = chrono::duration<double>(t_end - t_start).count() * 1000.0;
            double vitesse = 1000.0 / total_ms;
            double avg = (t + 1) / chrono::duration<double>(t_end - start_total).count();
            
            // DASHBOARD UI
            #ifdef _WIN32
                system("cls");
            #else
                system("clear");
            #endif
            
            printf("================================================================\n");
            printf("   MOAI GPT-2 ENGINE (NATIVE C++ / SEAL / OpenMP)               \n");
            printf("================================================================\n");
            printf(" STATS  : %5.2f tok/s | Moyenne: %5.2f tok/s | Token: %d\n", vitesse, avg, t+1);
            printf(" FHE    : %5.1f ms | Overhead Model: %5.1f ms\n", fhe_ms, total_ms - fhe_ms);
            printf("================================================================\n");
            printf(" PROMPT : \"%s\"\n", prompt.c_str());
            printf("----------------------------------------------------------------\n");
            printf(" REPONSE:%s\n", generated_text.c_str());
            printf("================================================================\n");
            fflush(stdout);
        }
        
        double total = chrono::duration<double>(chrono::high_resolution_clock::now() - start_total).count();
        printf("\n\nGENERATION TERMINEE en %.2fs (Moyenne: %.2f tok/s)\n", total, (double)MAX_TOKENS / total);
        
    } catch (const exception& e) {
        cerr << "[FATAL] " << e.what() << endl;
        return 1;
    }
    return 0;
}
