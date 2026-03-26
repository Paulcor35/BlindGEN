import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import struct
import os
import json
import argparse

def write_tensor(f, tensor, name=""):
    """Écrit un tenseur en binaire : [ndims][shape...][data float64]"""
    # Force Float64 pour la compatibilité CKKS SEAL
    arr = tensor.detach().float().cpu().numpy().astype(np.float64)
    dims = arr.shape
    f.write(struct.pack("i", len(dims)))
    for d in dims:
        f.write(struct.pack("i", d))
    f.write(arr.tobytes())
    size_mb = arr.nbytes / (1024*1024)
    print(f"  {name:40} | Shape: {str(dims):20} | {size_mb:6.2f} MB")

def get_model_mapper(model_type):
    """Retourne les chemins des poids pour chaque architecture"""
    mappers = {
        "gpt2": {
            "wte": "transformer.wte.weight",
            "wpe": "transformer.wpe.weight",
            "ln_f": "transformer.ln_f",
            "layers": "transformer.h",
            "type": "radford",
            "layer_naming": {
                "ln1": "ln_1",
                "ln2": "ln_2",
                "attn_qkv": "attn.c_attn",
                "attn_proj": "attn.c_proj",
                "mlp_up": "mlp.c_fc",
                "mlp_down": "mlp.c_proj"
            }
        },
        "phi3": {
            "wte": "model.embed_tokens.weight",
            "ln_f": "model.norm",
            "layers": "model.layers",
            "type": "llama",
            "layer_naming": {
                "ln1": "input_layernorm",
                "ln2": "post_attention_layernorm",
                "attn_q": "self_attn.q_proj",
                "attn_k": "self_attn.k_proj",
                "attn_v": "self_attn.v_proj",
                "attn_proj": "self_attn.o_proj",
                "mlp_up": "mlp.gate_up_proj", # Souvent combiné dans Phi-3
                "mlp_down": "mlp.down_proj"
            }
        }
    }
    # Fallback pour Qwen/StableLM (utilisent souvent le type llama)
    if "qwen" in model_type or "stablelm" in model_type:
        return mappers["phi3"]
    return mappers.get(model_type, mappers["gpt2"])

def export_universal(model_id, output_dir=None):
    if output_dir is None:
        output_dir = "moai_export_" + model_id.replace("/", "_")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Démarrage de l'exportation : {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
    model.eval()
    
    config = model.config
    m_type = config.model_type
    mapper = get_model_mapper(m_type)
    
    print(f"📦 Architecture détectée : {m_type} (Mapper: {mapper['type']})")
    
    # 1. Config JSON (Standardisée pour le C++)
    cfg_moai = {
        "model_id": model_id,
        "model_type": m_type,
        "n_layer": getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12)),
        "n_head": getattr(config, "n_head", getattr(config, "num_attention_heads", 12)),
        "n_embd": getattr(config, "n_embd", getattr(config, "hidden_size", 768)),
        "vocab_size": config.vocab_size,
        "activation": "GELU" if "gpt2" in m_type else "SILU",
        "layer_norm_eps": getattr(config, "layer_norm_epsilon", getattr(config, "rms_norm_eps", 1e-5))
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg_moai, f, indent=2)

    # 2. Export du Vocabulaire (Tokenizer C++)
    print("\n--- Tokenizer ---")
    export_tokenizer_vocab(tokenizer, os.path.join(output_dir, "vocab.txt"))

    # 3. Export des Poids (Binaire)
    weights_path = os.path.join(output_dir, "weights.bin")
    with open(weights_path, "wb") as f:
        print("\n--- Embeddings ---")
        write_tensor(f, model.get_parameter(mapper["wte"]), "Token Embeddings")
        if "wpe" in mapper:
            write_tensor(f, model.get_parameter(mapper["wpe"]), "Position Embeddings")
        
        # Layers
        layers = model.get_submodule(mapper["layers"])
        ln = mapper["layer_naming"]
        
        for i, block in enumerate(layers):
            print(f"\n--- Layer {i} ---")
            # LN 1
            write_tensor(f, block.get_submodule(ln["ln1"]).weight, f"L{i}.ln1.weight")
            if hasattr(block.get_submodule(ln["ln1"]), "bias") and block.get_submodule(ln["ln1"]).bias is not None:
                write_tensor(f, block.get_submodule(ln["ln1"]).bias, f"L{i}.ln1.bias")
            
            # Attention
            if "attn_qkv" in ln:
                write_tensor(f, block.get_submodule(ln["attn_qkv"]).weight, f"L{i}.attn.qkv.weight")
                if hasattr(block.get_submodule(ln["attn_qkv"]), "bias") and block.get_submodule(ln["attn_qkv"]).bias is not None:
                    write_tensor(f, block.get_submodule(ln["attn_qkv"]).bias, f"L{i}.attn.qkv.bias")
            elif "attn_q" in ln:
                # Modèles séparés (Q, K, V)
                for k in ["q", "k", "v"]:
                    write_tensor(f, block.get_submodule(ln[f"attn_{k}"]).weight, f"L{i}.attn.{k}.weight")
            
            write_tensor(f, block.get_submodule(ln["attn_proj"]).weight, f"L{i}.attn.proj.weight")
            
            # MLP : Support Dynamique Fusion gate_up (Standard MOAI)
            try:
                # Cas 1 : Couche déjà fusionnée (ex: Phi-3)
                mlp_up_weight = block.get_submodule(ln["mlp_up"]).weight
            except Exception:
                # Cas 2 : Couches séparées (ex: StableLM / Llama-2) -> On fusionne
                # On tente de trouver gate_proj et up_proj directement sur block.mlp
                gate_w = block.mlp.gate_proj.weight
                up_w = block.mlp.up_proj.weight
                mlp_up_weight = torch.cat([gate_w, up_w], dim=0)
                print(f"  [INFO] Fusion gate_proj + up_proj pour L{i}")

            write_tensor(f, mlp_up_weight, f"L{i}.mlp.up.weight")
            
            # MLP Down
            try:
                mlp_down_w = block.get_submodule(ln["mlp_down"]).weight
            except Exception:
                mlp_down_w = block.mlp.down_proj.weight
            write_tensor(f, mlp_down_w, f"L{i}.mlp.down.weight")

        # Final
        print(f"\n--- Output Head ---")
        write_tensor(f, model.get_submodule(mapper["ln_f"]).weight, "Final Norm Weight")
        write_tensor(f, model.lm_head.weight, "LM Head Weight")

    print(f"\n✅ Terminé ! Modèle exporté dans {output_dir}")

def export_tokenizer_vocab(tokenizer, vocab_path):
    """Exporte le vocabulaire pour le décodage C++ (Support Hexa pour GPT-2/BPE)"""
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    vocab = tokenizer.get_vocab()
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            # Détection BPE Style (GPT-2)
            if any(c in token for c in byte_decoder):
                raw_bytes = bytes([byte_decoder.get(c, 0) for c in token])
            else:
                # Modèles standard (Phi-3, Qwen)
                raw_bytes = token.encode("utf-8")
            
            token_hex = raw_bytes.hex()
            f.write(f"{token_id}\t{token_hex}\n")
    print(f"  Vocabulaire sauvegardé en {vocab_path} ({len(vocab)} tokens).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    export_universal(args.model, args.out)
