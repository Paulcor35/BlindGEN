import time
import torch
import tenseal as ts
import numpy as np
from transformers import AutoTokenizer, DistilBertModel

# On importe la méthode compacte de ton collègue (qui est dans le même dossier)
from engines.compact_method import CompactActivation

class BlindSDK:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = DistilBertModel.from_pretrained(model_name).embeddings
        self.embedding_model.eval()
        
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 30, 30, 30, 30, 30, 30, 60]
        )
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        self.context.global_scale = 2 ** 30
        
    def encrypt_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.embedding_model(inputs['input_ids'])
            
        encrypted_vectors = []
        for i in range(embeddings.shape[1]):
            vec = embeddings[0, i, :].numpy().tolist()
            enc_vec = ts.ckks_vector(self.context, vec)
            encrypted_vectors.append(enc_vec)
            
        return encrypted_vectors, self.context.serialize()

class BlindServer:
    def __init__(self, model_name='distilbert-base-uncased'):
        full_model = DistilBertModel.from_pretrained(model_name)
        self.layer = full_model.transformer.layer[0] 
        self.layer.eval()
        
        self.compact_gelu = CompactActivation(func_type='gelu', m=1, k=4, range_val=(-4, 4))
        self.gelu_coeffs = self.compact_gelu.pieces[0]['poly'].convert(kind=np.polynomial.Polynomial).coef.tolist()
        
    def process_request(self, encrypted_vectors, context_serialized):
        server_context = ts.context_from(context_serialized)
        
        W_q = self.layer.attention.q_lin.weight.detach().numpy().T
        b_q = self.layer.attention.q_lin.bias.detach().numpy()
        
        processed_Q = []
        for enc_vec in encrypted_vectors:
            q_i = enc_vec.matmul(W_q.tolist()) + b_q.tolist()
            processed_Q.append(q_i)
            
        W_ff1 = self.layer.ffn.lin1.weight.detach().numpy().T[:768, :16] 
        b_ff1 = self.layer.ffn.lin1.bias.detach().numpy()[:16]
        
        final_results = []
        for q_i in processed_Q:
            ff1_i = q_i.matmul(W_ff1.tolist()) + b_ff1.tolist()
            ff1_activated = ff1_i.polyval(self.gelu_coeffs)
            final_results.append(ff1_activated)
            
        return final_results

# --- LA FONCTION QUE TON INTERFACE STREAMLIT VA APPELER ---
def run_compact_pipeline(texte, sdk, server):
    """
    Exécute l'ensemble du processus pour la méthode Compact.
    Retourne : (échantillon_chiffré, temps_execution, description)
    """
    start_time = time.time()
    
    # 1. Le Client chiffre
    encrypted_vectors, context_ser = sdk.encrypt_text(texte)
    
    # Pour l'affichage UI, on prend un bout du vecteur chiffré (pour prouver que c'est illisible)
    sample_hex = encrypted_vectors[0].serialize()[:30].hex()
    ciphertext_display = f"[0x{sample_hex}...]"
    
    # 2. Le Serveur calcule
    results = server.process_request(encrypted_vectors, context_ser)
    
    # 3. Le Client déchiffre (pour valider)
    decrypted_sample = results[0].decrypt(sdk.context.secret_key())
    
    end_time = time.time()
    duree = round(end_time - start_time, 2)
    description = f"Activation GELU via Polynômes de Tchebychev (PoPETS 2024). Extrait déchiffré: {decrypted_sample[:3]}"
    
    return ciphertext_display, duree, description