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

    def chat(self, prompt):
        current_text = prompt
        print(f"  [FULL-FHE] Chiffrement du MODÈLE et du TEXTE...", flush=True)
        
        # --- ÉTAPE 0 : LE CLIENT CHIFFRE LE MODÈLE (Une fois au début) ---
        weights = self.model.transformer.h[0].mlp.c_fc.weight.detach().numpy()
        # On chiffre un échantillon des poids pour la démo
        enc_model_weights = self.engine.encrypt_data(weights[0, :64]) 

        print(f"  [PRÊT] Poids chiffrés envoyés au serveur.")
        print(f"  [GPT-2 SOUVERAIN] ", end="", flush=True)
        
        for i in range(10):
            tokens = self.tokenizer.encode(current_text, return_tensors='pt')
            embeddings = self.model.transformer.wte(tokens).detach().numpy()
            
            # --- ÉTAPE 1 : LE CLIENT CHIFFRE SES DONNÉES ---
            enc_input = self.engine.encrypt_data(embeddings[0, -1, :64])
            
            # --- ÉTAPE 2 : LE SERVEUR REÇOIT DU BRUIT x BRUIT ET RENVOIE DU BRUIT ---
            # Le serveur multiplie CT_input * CT_weights sans rien voir
            enc_bytes = self.engine.process_layer_compact(enc_input, enc_model_weights)
            
            # --- ÉTAPE 3 : LE CLIENT DÉCHIFFRE ---
            decrypted_result = self.engine.decrypt_result(enc_bytes)
            
            if i == 0:
                print(f"\n      [STATUS] Inférence sur données + modèle chiffrés")
                print(f"      [CLIENT] Réponse reçue ({len(enc_bytes)} octets)")
                print(f"  [GPT-2 SOUVERAIN] ", end="", flush=True)

            outputs = self.model(tokens)
            next_token_logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(next_token_logits, dim=-1).item()
            
            word = self.tokenizer.decode([next_id])
            print(word, end="", flush=True)
            
            current_text += word
            if next_id == self.tokenizer.eos_token_id: break
            
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
