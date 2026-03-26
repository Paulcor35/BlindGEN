"""
Exporteur complet GPT-2 pour le moteur natif C++.
Exporte TOUS les poids nécessaires à l'inférence complète.
"""
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import struct
import os
import json

def write_tensor(f, tensor, name=""):
    """Écrit un tenseur en binaire : [ndims][shape...][data float64]"""
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    dims = arr.shape
    f.write(struct.pack("i", len(dims)))
    for d in dims:
        f.write(struct.pack("i", d))
    f.write(arr.tobytes())
    size_mb = arr.nbytes / (1024*1024)
    print(f"  {name}: shape={dims}, {size_mb:.2f} MB")

def export_gpt2_full(output_dir="gpt2_native"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Chargement du modèle GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    config = model.config
    print(f"Config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}, vocab={config.vocab_size}")
    
    # 1. Export de la config
    cfg = {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "vocab_size": config.vocab_size,
        "n_positions": config.n_positions,
        "layer_norm_epsilon": config.layer_norm_epsilon,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config sauvegardée.")
    
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    # 2. Export du vocabulaire (pour le tokenizer C++) en format raw bytes hex
    vocab = tokenizer.get_vocab()
    vocab_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            # Reconstituer les octets originaux (résout les problèmes d'encodage)
            raw_bytes = bytes([byte_decoder.get(c, 0) for c in token])
            token_hex = raw_bytes.hex()
            f.write(f"{token_id}\t{token_hex}\n")
            
    print(f"Vocabulaire sauvegardé en {vocab_path} ({len(vocab)} tokens).")
    
    # 3. Export des poids du modèle
    weights_path = os.path.join(output_dir, "weights.bin")
    total_bytes = 0
    with open(weights_path, "wb") as f:
        # --- Embeddings ---
        print("\n--- Embeddings ---")
        write_tensor(f, model.transformer.wte.weight, "wte (token embeddings)")
        write_tensor(f, model.transformer.wpe.weight, "wpe (position embeddings)")
        
        # --- Transformer Layers ---
        for i, block in enumerate(model.transformer.h):
            print(f"\n--- Layer {i} ---")
            # LayerNorm 1
            write_tensor(f, block.ln_1.weight, f"L{i}.ln1.weight")
            write_tensor(f, block.ln_1.bias, f"L{i}.ln1.bias")
            
            # Attention (c_attn = combined QKV projection, c_proj = output projection)
            write_tensor(f, block.attn.c_attn.weight, f"L{i}.attn.c_attn.weight")
            write_tensor(f, block.attn.c_attn.bias, f"L{i}.attn.c_attn.bias")
            write_tensor(f, block.attn.c_proj.weight, f"L{i}.attn.c_proj.weight")
            write_tensor(f, block.attn.c_proj.bias, f"L{i}.attn.c_proj.bias")
            
            # LayerNorm 2
            write_tensor(f, block.ln_2.weight, f"L{i}.ln2.weight")
            write_tensor(f, block.ln_2.bias, f"L{i}.ln2.bias")
            
            # MLP (c_fc = up projection, c_proj = down projection)
            write_tensor(f, block.mlp.c_fc.weight, f"L{i}.mlp.c_fc.weight")
            write_tensor(f, block.mlp.c_fc.bias, f"L{i}.mlp.c_fc.bias")
            write_tensor(f, block.mlp.c_proj.weight, f"L{i}.mlp.c_proj.weight")
            write_tensor(f, block.mlp.c_proj.bias, f"L{i}.mlp.c_proj.bias")
        
        # --- Final LayerNorm ---
        print(f"\n--- Final LN ---")
        write_tensor(f, model.transformer.ln_f.weight, "ln_f.weight")
        write_tensor(f, model.transformer.ln_f.bias, "ln_f.bias")
        
        # --- LM Head (tied with wte in GPT-2, mais on l'écrit quand même pour simplifier le C++) ---
        print(f"\n--- LM Head ---")
        write_tensor(f, model.lm_head.weight, "lm_head.weight")
    
    total_size = os.path.getsize(weights_path)
    print(f"\n{'='*60}")
    print(f"Export terminé! {total_size / (1024*1024):.1f} MB -> {weights_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    export_gpt2_full()
