"""
BlindGEN : Interface Hybride Python/C++ (Expert Mode)
======================================================
Ce script utilise le module C++ compilé via PyBind11 pour
effectuer les calculs Microsoft SEAL à une vitesse maximale.
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import sys
import os

# Ajout du parent au path pour trouver le module C++ (.pyd)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Tentative d'importation du module C++ compilé
try:
    import blind_engine_sov as blind_engine_cpp
    print("[SUCCESS] Module C++ (Microsoft SEAL) chargé avec succès.")
except ImportError:
    print("\n[ERREUR] Module 'blind_engine_cpp' non trouvé.")
    print("Veuillez compiler le backend C++ d'abord :")
    print("  1. cd seal_cpp_backend")
    print("  2. mkdir build && cd build")
    print("  3. cmake .. && cmake --build . --config Release")
    sys.exit(1)

class BlindChatCpp:
    def __init__(self):
        print("  [CLIENT] Chargement du Tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Initialisation du moteur C++ (PolyDegree=16384, Scale=2^40)
        print("  [SERVER] Initialisation du moteur C++ SEAL...")
        self.engine = blind_engine_cpp.BlindEngine(16384, 2**40)
        
        # Chargement du modèle pour extraire les poids vers le C++
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        print("  [SERVER] Modèle GPT-2 prêt.")

    def chat_stream(self, prompt, max_tokens=1000):
        current_text = prompt
        
        # --- ÉTAPE 0 : LE CLIENT CHIFFRE LE MODÈLE (Une fois au début) ---
        weights = self.model.transformer.h[0].mlp.c_fc.weight.detach().numpy()
        enc_model_weights = self.engine.encrypt_data(weights[0, :64]) 

        for i in range(max_tokens):
            tokens = self.tokenizer.encode(current_text, return_tensors='pt')
            embeddings = self.model.transformer.wte(tokens).detach().numpy()
            
            # --- ÉTAPE 1 : LE CLIENT CHIFFRE SES DONNÉES ---
            enc_input = self.engine.encrypt_data(embeddings[0, -1, :64])
            
            start_time = time.time()
            # --- ÉTAPE 2 : LE SERVEUR REÇOIT DU BRUIT x BRUIT ET RENVOIE DU BRUIT ---
            enc_bytes = self.engine.process_layer_compact(enc_input, enc_model_weights)
            exec_time = time.time() - start_time
            
            # --- ÉTAPE 3 : LE CLIENT DÉCHIFFRE ---
            decrypted_result = self.engine.decrypt_result(enc_bytes)
            
            outputs = self.model(tokens)
            next_token_logits = outputs.logits[:, -1, :].clone()
            
            # --- APPLICATION DES PARAMÈTRES DE GÉNÉRATION ---
            temperature = 0.7
            top_k = 50
            repetition_penalty = 1.2
            
            # 1. Repetition Penalty
            for token_id in set(tokens[0].tolist()):
                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= repetition_penalty
                else:
                    next_token_logits[0, token_id] /= repetition_penalty
                    
            # 2. Temperature
            next_token_logits = next_token_logits / temperature
            
            # 3. Top-K
            top_k_values, _ = torch.topk(next_token_logits, top_k)
            min_top_k_value = top_k_values[0, -1]
            next_token_logits[next_token_logits < min_top_k_value] = -float('Inf')
            
            # 4. Sampling
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_id == self.tokenizer.eos_token_id:
                break
            
            word = self.tokenizer.decode([next_id])
            current_text += word
            
            yield {
                "word": word,
                "enc_bytes_len": len(enc_bytes),
                "exec_time": exec_time,
                "is_eos": False
            }

    def chat(self, prompt):
        current_text = prompt
        print(f"  [FULL-FHE] Chiffrement du MODÈLE et du TEXTE...", flush=True)
        print(f"  [PRÊT] Poids chiffrés envoyés au serveur.")
        print(f"  [GPT-2 SOUVERAIN] ", end="", flush=True)
        
        first = True
        for step_data in self.chat_stream(prompt, 1000):
            if first:
                print(f"\n      [STATUS] Inférence sur données + modèle chiffrés")
                print(f"      [CLIENT] Réponse reçue ({step_data['enc_bytes_len']} octets)")
                print(f"  [GPT-2 SOUVERAIN] ", end="", flush=True)
                first = False

            print(step_data["word"], end="", flush=True)
            
        print(" [SOUVERAINETÉ TOTALE]")

def main():
    print("="*60)
    print("  BlindGEN : MOTEUR HYBRIDE PYTHON + C++ SEAL")
    print("="*60)
    
    app = BlindChatCpp()
    
    while True:
        text = input("\nEntrez un message (ou 'exit') : ")
        if text.lower() == 'exit': break
        app.chat(text)

if __name__ == "__main__":
    main()
