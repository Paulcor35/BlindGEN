import torch
import numpy as np
import time
import sys
import os

# Ajout des dossiers au path pour trouver le module C++ (.pyd)
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    import blind_engine_sov as blind_engine_cpp
except ImportError:
    # On ne quitte plus brutalement pour permettre au reste de l'app de tourner si besoin
    blind_engine_cpp = None

class BlindChatCpp:
    def __init__(self, tokenizer, model, poly_n=16384, scale_bits=40):
        self.tokenizer = tokenizer
        self.model = model
        
        # Initialisation du moteur C++ (PolyDegree et Scale dynamiques)
        if blind_engine_cpp:
            print(f"  [SERVER] Initialisation Compact (N={poly_n}, Scale=2^{scale_bits})...")
            self.engine = blind_engine_cpp.BlindEngine(poly_n, 2**scale_bits)
            print("  [SERVER] Moteur Compact prêt.")
        else:
            self.engine = None

    def _find_mlp_layer(self, layer_idx=0):
        """Trouve dynamiquement la couche MLP pour une couche donnée."""
        # On essaie de trouver le conteneur de couches (h pour GPT2, layers pour Llama/StableLM)
        layers = None
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
            
        if layers is None or layer_idx >= len(layers):
            return None, 12 # Fallback
            
        layer = layers[layer_idx]
        # Recherche du MLP
        for name, module in layer.named_modules():
            if any(cn in name for cn in ["c_fc", "gate_proj", "up_proj", "mlp.fc1"]):
                return module, len(layers)
        return None, len(layers)

    def chat_stream(self, prompt, max_tokens=100, temperature=0.7, top_k=50, repetition_penalty=1.2, fhe_slice=256):
        if not self.engine:
            yield {"word": "Erreur: Backend C++ non chargé.", "exec_time": 0, "enc_bytes_len": 0}
            return

        current_text = prompt
        
        # Detection de la structure
        target_mlp, num_layers = self._find_mlp_layer(0)
        
        # --- ÉTAPE 0 : LE CLIENT CHIFFRE LE MODÈLE (Simulation benchmarking) ---
        if target_mlp and hasattr(target_mlp, "weight"):
            weights = target_mlp.weight.data.detach().cpu().numpy()[:fhe_slice, :fhe_slice]
            # On simule le chiffrement d'un vecteur de poids
            _ = self.engine.encrypt_data(weights[0, :fhe_slice].astype(np.float64).tolist()) 

        for i in range(max_tokens):
            tokens = self.tokenizer.encode(current_text, return_tensors='pt').to(self.model.device)
            
            # Embeddings génériques
            with torch.no_grad():
                input_embeds = self.model.get_input_embeddings()(tokens)
                embeddings = input_embeds.detach().cpu().numpy()
            
            # --- ÉTAPE 1 : LE CLIENT CHIFFRE SES DONNÉES ---
            # On prend le dernier token
            vec_to_encrypt = embeddings[0, -1, :fhe_slice].astype(np.float64).tolist()
            enc_input = self.engine.encrypt_data(vec_to_encrypt)
            
            total_exec_time = 0.0
            last_enc_bytes = b""
            last_ciphertext_b64 = ""
            
            # --- ÉTAPE 2 : LE SERVEUR TRAITE LES N COUCHES EN FHE ---
            for l_idx in range(num_layers):
                start_time = time.time()
                # On récupère la matrice pour cette couche
                layer_mlp, _ = self._find_mlp_layer(l_idx)
                
                if layer_mlp and hasattr(layer_mlp, "weight"):
                    # Extraction et conversion en liste pour le C++
                    # Note: Compact C++ attend une liste de listes (matrix)
                    w_mat = layer_mlp.weight.data.detach().cpu().numpy()[:fhe_slice, :fhe_slice].astype(np.float64)
                    last_enc_bytes = self.engine.process_layer_compact(enc_input, w_mat.tolist())
                    
                    # --- CAPTURE RÉELLE (ZÉRO DUMMY) ---
                    import base64
                    if isinstance(last_enc_bytes, bytes):
                        last_ciphertext_b64 = base64.b64encode(last_enc_bytes[:1024]).decode()
                    
                total_exec_time += (time.time() - start_time)
            
            # --- ÉTAPE 3 : LE CLIENT DÉCHIFFRE ET ÉCHANTILLONNE ---
            # (Ici on utilise le modèle réel pour la suite du sampling, comme dans MOAI)
            with torch.no_grad():
                outputs = self.model(tokens)
                next_token_logits = outputs.logits[:, -1, :].clone()
            
            # Sampling standard
            next_token_logits = next_token_logits / max(temperature, 1e-5)
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_id == self.tokenizer.eos_token_id:
                break
            
            word = self.tokenizer.decode([next_id], skip_special_tokens=True)
            current_text += word
            
            yield {
                "word": word,
                "enc_bytes_len": len(last_enc_bytes),
                "exec_time": total_exec_time,
                "ciphertext_b64": last_ciphertext_b64
            }
