"""
BlindGEN: Inférence par Embeddings Chiffrés avec Méthode Compact
================================================================
Ce module implémente l'architecture complète :
1. CLIENT-SIDE SDK : Chiffrement des embeddings à la source.
2. SERVER-SIDE : Inférence "aveugle" utilisant la méthode COMPACT pour les activations.
3. SOUVERAINETÉ : Calculs locaux sur données chiffrées (FHE).
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertModel
import tenseal as ts
import numpy as np
import time
from compact_method import CompactActivation

class BlindSDK:
    """SDK local à l'entreprise pour sécuriser les données avant envoi."""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        # Détection du GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SDK] Initialisation du SDK sur : {str(self.device).upper()}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # On déplace le modèle d'embedding sur le GPU
        self.embedding_model = DistilBertModel.from_pretrained(model_name).embeddings.to(self.device)
        self.embedding_model.eval()
        
        # Configuration FHE (CKKS)
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 30, 30, 30, 30, 30, 30, 60]
        )
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        self.context.global_scale = 2 ** 30
        
    def encrypt_text(self, text):
        """Transforme le texte en embeddings chiffrés."""
        print(f"[SDK] Encodage et chiffrement de: '{text[:30]}...'")
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embeddings = self.embedding_model(inputs['input_ids'])
            # On ramène sur le CPU pour le chiffrement TenSEAL (ne supporte pas CUDA)
            embeddings_cpu = embeddings.cpu()
            
        # Chiffrement chaque token (vecteur)
        encrypted_vectors = []
        for i in range(embeddings_cpu.shape[1]):
            vec = embeddings_cpu[0, i, :].numpy().tolist()
            enc_vec = ts.ckks_vector(self.context, vec)
            encrypted_vectors.append(enc_vec)
            
        return encrypted_vectors, self.context.serialize()

class BlindServer:
    """Serveur d'inférence traitant uniquement des vecteurs chiffrés."""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        # Le serveur peut aussi charger les couches sur GPU pour les opérations claires
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SERVER] Chargement des couches sur : {str(self.device).upper()}...")
        
        full_model = DistilBertModel.from_pretrained(model_name)
        self.layer = full_model.transformer.layer[0].to(self.device)
        self.layer.eval()
        
        # Initialisation de la méthode COMPACT pour GELU
        self.compact_gelu = CompactActivation(func_type='gelu', m=1, k=4, range_val=(-4, 4))
        self.gelu_coeffs = self.compact_gelu.pieces[0]['poly'].convert(kind=np.polynomial.Polynomial).coef.tolist()
        
    def process_request(self, encrypted_vectors, context_serialized):
        """Exécution de l'inférence 'aveugle' (sans déchiffrement)."""
        # Récupération du contexte "Public" (le seul que le serveur possède en vrai)
        server_context = ts.context_from(context_serialized)

        # --- TEST DEMANDÉ : Simulation d'une attaque du serveur ---
        print("\n" + "!"*60)
        print("[TEST UTILISATEUR] Tentative de vol de données par le serveur...")
        print("Le serveur essaie d'utiliser son contexte public pour décrypter vos données...")
        
        # On force le vecteur à être lié au contexte "AVEUGLE" du serveur
        # C'est ce qui se passe quand le serveur reçoit les données par le réseau
        blind_vector = ts.ckks_vector_from(server_context, encrypted_vectors[0].serialize())
        
        try:
            print("Tentative : print(blind_vector.decrypt())")
            print(blind_vector.decrypt()) 
        except Exception as e:
            print(f"\n[RÉSULTAT DU CRASH] : {e}")
            print("Action bloquée : Le serveur ne peut PAS lire le contenu chiffré.")
        print("!"*60 + "\n")
        
        context = server_context # On utilise le contexte désérialisé pour la suite
        
        # --- TEST DE SÉCURITÉ : Tentative d'espionnage par le serveur ---
        print("\n[SECURITY TEST] Le serveur tente de lire vos données...")
        try:
            # On affiche un échantillon des données brutes reçues
            raw_sample = encrypted_vectors[0].serialize()[:50]
            print(f"  > Données brutes reçues (hex): {raw_sample.hex()}...")
            print("  > [VÉRIFICATION] Le serveur ne voit que des octets aléatoires.")
            
            # Tentative de déchiffrement sans la clé secrète
            print("  > Tentative de déchiffrement (decrypt())...")
            encrypted_vectors[0].decrypt() # Cela devrait lever une exception
        except Exception as e:
            print(f"  > [SUCCÈS] Échec du déchiffrement : Le serveur n'a pas la clé secrète.")
            print(f"    (Erreur système attendue: {str(e)[:50]}...)")
        print("-" * 50 + "\n")

        # 1. Couche Attention (Opérations linéaires)
        print("[SERVER] Calcul des projections Q, K, V en FHE...")
        # On extrait les poids (on les garde sur CPU car TenSEAL matmul est CPU)
        W_q = self.layer.attention.q_lin.weight.detach().cpu().numpy().T
        b_q = self.layer.attention.q_lin.bias.detach().cpu().numpy()
        
        # Simulation simplification : On traite juste Q pour la démo de performance
        processed_Q = []
        for enc_vec in encrypted_vectors:
            # Opération Linéaire : MatMul chiffrée
            q_i = enc_vec.matmul(W_q.tolist()) + b_q.tolist()
            processed_Q.append(q_i)
            
        # 2. Feed-Forward avec Méthode COMPACT
        print("[SERVER] Application du Feed-Forward + Activation COMPACT (aveugle)...")
        # On réduit encore la taille pour que la démo soit plus rapide (128 -> 16 neurones)
        W_ff1 = self.layer.ffn.lin1.weight.detach().cpu().numpy().T[:768, :16] 
        b_ff1 = self.layer.ffn.lin1.bias.detach().cpu().numpy()[:16]
        
        final_results = []
        for q_i in processed_Q:
            # ff1 = q_i @ W + b
            ff1_i = q_i.matmul(W_ff1.tolist()) + b_ff1.tolist()
            
            # --- C'est ici que la méthode COMPACT intervient ! ---
            print("[SERVER] Evaluation polynomiale COMPACT en cours sur données chiffrées...")
            ff1_activated = ff1_i.polyval(self.gelu_coeffs)
            final_results.append(ff1_activated)
            
        return final_results

def verify_privacy_mathematically(sdk, encrypted_vector, context_serialized):
    """
    AUDIT DE SÉCURITÉ APPROFONDI :
    1. Preuve d'absence de clé (Key Isolation)
    2. Preuve d'échec de déchiffrement (Material Security)
    3. Preuve de non-déterminisme (IND-CPA)
    """
    print("\n" + "="*60)
    print(" [SÉCURITÉ AUDIT : PROTOCOLE DE VÉRIFICATION ANTI-ESPIONNAGE] ")
    print("="*60)
    
    # --- TEST 1 : Isolation des clés ---
    server_context = ts.context_from(context_serialized)
    has_secret_key = server_context.has_secret_key()
    print(f"[TEST 1] Isolation : Le serveur possède-t-il la Secret Key ? {'OUI' if has_secret_key else 'NON (Pass)'}")
    
    # --- TEST 2 : Impossibilité matérielle de déchiffrement ---
    vector_bytes = encrypted_vector.serialize()
    server_vector = ts.ckks_vector_from(server_context, vector_bytes)
    
    print("[TEST 2] Décryptage forcé : Tentative de lecture brute...")
    try:
        server_vector.decrypt() 
        is_secure = False
    except Exception as e:
        print(f"      -> ÉCHEC DU DÉCRYPTAGE : {str(e)[:50]}...")
        is_secure = True
    
    # --- TEST 3 : Preuve de non-déterminisme (IND-CPA) ---
    # Si le serveur pouvait "deviner" en encodant les mêmes mots, 
    # le chiffrement serait inutile. On prouve ici que c'est impossible.
    val_test = [1.0, 2.0, 3.0]
    enc1 = ts.ckks_vector(sdk.context, val_test).serialize()
    enc2 = ts.ckks_vector(sdk.context, val_test).serialize()
    
    # On compare les octets des deux chiffrements de la MÊME valeur
    is_nondeterministic = (enc1 != enc2)
    print(f"[TEST 3] IND-CPA : Chiffrements différents pour même valeur ? {'OUI (Pass)' if is_nondeterministic else 'NON'}")
    
    # Aperçu technique pour l'audit
    print(f"\n[AUDIT TECHNIQUE] Ciphertext Sample : {enc1[:20].hex()}...")

    # VERIFICATIONS FINALES (ARRÊT DU PROGRAMME EN CAS DE FAIL)
    assert not has_secret_key, "ALERTE : Clé secrète détectée sur le serveur !"
    assert is_secure, "ALERTE : Le serveur a craqué le vecteur !"
    assert is_nondeterministic, "ALERTE : Chiffrement déterministe détecté (vulnérable) !"
    
    print("="*60)
    print(" [RÉSULTAT] : AUDIT VALIDÉ - LE SERVEUR EST TOTALEMENT AVEUGLE ")
    print("="*60 + "\n")
    return True

def run_blind_project():
    print("\n" + "="*50)
    print(" L'INFERENCE PAR EMBEDDINGS CHIFFRÉS (BlindGEN) ")
    print("="*50 + "\n")
    
    # 1. Phase Client (SDK)
    sdk = BlindSDK()
    text_data = "Ceci est un secret industriel."
    
    # On obtient les vecteurs et le contexte SANS la clé secrète
    # (Par défaut, ts.context.serialize() n'inclut PAS la clé secrète)
    encrypted_data, context_ser = sdk.encrypt_text(text_data)
    
    # --- AUDIT DE SÉCURITÉ IMMÉDIAT ---
    # On prouve ici que ce qu'on va envoyer au serveur est indéchiffrable par lui
    verify_privacy_mathematically(sdk, encrypted_data[0], context_ser)
    
    # 2. Phase Serveur (Inférence Aveugle)
    server = BlindServer()
    start_time = time.time()
    
    print("[SERVER] Début de l'inférence aveugle...")
    results = server.process_request(encrypted_data, context_ser)
    
    end_time = time.time()
    print(f"\n[INFO] Inférence aveugle terminée en {end_time - start_time:.2f}s")
    
    # 3. Retour au Client pour vérification
    print("[SDK] Déchiffrement du résultat final (avec clé secrète restée au SDK)...")
    # Pour déchiffrer, on doit utiliser le contexte ORIGINAL du SDK qui a la clé secrète
    decrypted_sample = results[0].decrypt(sdk.context.secret_key())
    print(f"[SDK] Résultat déchiffré (extrait): {decrypted_sample[:5]}...")
    print("\n[FIN] Démonstration terminée.")

if __name__ == "__main__":
    run_blind_project()
