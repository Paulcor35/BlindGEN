import time
import sys
import os
import yaml
import gc
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from moai_paper_implementation import MOAIPaperCKKS, MOAILayerNorm

FHE_SLICE_SIZE = 256 # Taille du benchmark FHE (fragment du vecteur caché)

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

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

def find_fhe_benchmark_layer(model):
    candidates = ["c_fc", "fc1", "gate_proj", "up_proj", "mlp"]
    for name, module in model.named_modules():
        mod_type = module.__class__.__name__
        if any(c in name.lower() for c in candidates):
            if "Linear" in mod_type or "Conv1D" in mod_type:
                if hasattr(module, "weight") and module.weight is not None:
                    sh = module.weight.shape
                    is_expansion = (sh[1] > sh[0]) if "Conv1D" in mod_type else (sh[0] > sh[1])
                    if is_expansion or "mlp" in name.lower():
                        print(f"[MOAI-FHE] Couche cible détectée : {name} ({mod_type})")
                        return module
    return None

def generate_realtime_dashboard(model, tokenizer, prompt, max_tokens=100, 
                               temperature=0.7, top_k=50, repetition_penalty=1.2, 
                               model_name="LLM", use_fhe=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    
    fhe_engine, W_fhe, B_fhe, target_layer = None, None, None, None
    if use_fhe:
        print("[SERVER] Initialisation du moteur FHE C++ SEAL...")
        fhe_engine = MOAIPaperCKKS()
        target_layer = find_fhe_benchmark_layer(model)
        if target_layer:
            # Correction: .float() avant .cpu().numpy() pour supporter BFloat16 (StableLM/Phi3)
            W_fhe = target_layer.weight.data.float().cpu().numpy()[:FHE_SLICE_SIZE, :FHE_SLICE_SIZE].astype(np.float64)
            if target_layer.bias is not None:
                B_fhe = target_layer.bias.data.float().cpu().numpy()[:FHE_SLICE_SIZE].astype(np.float64)
            else:
                B_fhe = np.zeros(FHE_SLICE_SIZE)
        else:
            print("ERREUR: Impossible de trouver une couche MLP pour le benchmark FHE.")
            sys.exit(1)
    
    generated_text, t_start = "", time.time()
    cls()
    
    for i in range(max_tokens):
        t_token_start = time.time()
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            fhe_calc_time = 0
            if use_fhe and target_layer:
                t_fhe_s = time.time()
                # Correction: .float() ici aussi pour le hidden state
                h_state = outputs.hidden_states[1][0, -1, :FHE_SLICE_SIZE].float().cpu().numpy().reshape(1, -1).astype(np.float64)
                enc_h = fhe_engine.col_pack_encrypt(h_state)
                _ = fhe_engine.he_cpmm(enc_h, W_fhe, B_fhe, fhe_engine.public_ctx, server=fhe_engine.server)
                fhe_calc_time = (time.time() - t_fhe_s) * 1000.0
                del enc_h
                gc.collect()

            logits = outputs.logits[:, -1, :]
            for token_id in set(input_ids[0].tolist()):
                val = logits[0, token_id]
                logits[0, token_id] = val / repetition_penalty if val > 0 else val * repetition_penalty

            if temperature > 0: logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id: break
            generated_text += tokenizer.decode(next_token[0])
            
            t_total_ms = (time.time() - t_token_start) * 1000.0
            avg_v = (i + 1) / (time.time() - t_start)
            sys.stdout.write("\033[H") 
            print("="*80)
            print(f" MOAI FHE BENCHMARK : {model_name}")
            print(f" STATS: {avg_v:6.2f} tok/s | Token: {i+1}")
            print(f" LATENCE FHE (C++ CKKS): {fhe_calc_time:6.1f} ms")
            print(f" OVERHEAD (Sampling):    {t_total_ms - fhe_calc_time:6.1f} ms")
            print("="*80)
            print(f" RÉPONSE: {generated_text}")
            sys.stdout.write("\033[J") 
            sys.stdout.flush()
    print(f"\nTERMINÉ en {time.time() - t_start:.2f}s")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MOAI Universal FHE Checker")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--fhe", action="store_true", default=True)
    parser.add_argument("--no-fhe", action="store_false", dest="fhe")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--penalty", type=float, default=1.2)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "moai_config.yaml")
    with open(config_path, "r") as f: config = yaml.safe_load(f)

    print(f"Chargement de {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=True)
    model = inject_moai_engine(model, args.model, config)
    model.eval()
    generate_realtime_dashboard(model, tokenizer, args.prompt, max_tokens=args.tokens, model_name=args.model, use_fhe=args.fhe)

if __name__ == "__main__":
    main()
