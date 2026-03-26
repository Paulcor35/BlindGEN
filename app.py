import streamlit as st
import time
import sys
import os

# Ajout du path pour trouver les modules C++ (.pyd) et les dossiers méthodes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# On importe les moteurs FHE
from compact_method.blind_chat_cpp import BlindChatCpp, MAX_TOKENS
from moai_method.blind_chat_moai import BlindChatMoai

from transformers import GPT2Tokenizer, GPT2LMHeadModel

st.set_page_config(layout="wide", page_title="BlindGEN - Inférence Souveraine", page_icon="🔒")

# --- MISE EN CACHE DES MOTEURS FHE (chargés une seule fois) ---
@st.cache_resource
def init_engines():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt2_local")
    
    # 1. Chargement unique du modèle et du tokenizer
    print(f"  [GLOBAL] Chargement du modèle GPT-2 depuis {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    
    # 2. Partage avec les différentes méthodes FHE
    return {
        "Compact (PoPETS 2024)": BlindChatCpp(tokenizer, model),
        "MOAI": BlindChatMoai(tokenizer, model)
    }

engines = init_engines()

# --- BARRE LATÉRALE : SÉLECTION DE LA MÉTHODE ---
st.sidebar.title("Banc d'essai FHE")
st.sidebar.markdown("Sélectionnez l'architecture papier à évaluer :")

choix_methode = st.sidebar.radio(
    "Architecture d'inférence aveugle :",
    ["Compact (PoPETS 2024)", "MOAI"]
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Paramètres GPT-2")
max_tokens = st.sidebar.slider("Max Tokens", 10, 256, 100)
temperature = st.sidebar.slider("Température", 0.1, 2.0, 0.7)
top_k = st.sidebar.slider("Top-K", 1, 100, 50)
penalty = st.sidebar.slider("Pénalité de répétition", 1.0, 2.0, 1.2)

# --- INITIALISATION DE LA MÉMOIRE DU CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_logs" not in st.session_state:
    st.session_state.server_logs = []

# --- INTERFACE PRINCIPALE ---
st.title("Inférence Sécurisée - Benchmark des Architectures")
st.markdown("Démonstration en direct du traitement d'embeddings chiffrés via Microsoft SEAL (C++).")
st.divider()

col_user, col_server = st.columns(2)

# --- COLONNE GAUCHE : CLIENT ---
with col_user:
    st.subheader("👤 Vue Client (Local)")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    prompt = st.chat_input("Envoyer un texte à analyser en toute sécurité...")

# --- COLONNE DROITE : SERVEUR ---
with col_server:
    st.subheader("🖥️ Vue Serveur (Cloud)")
    st.info("Ce que le serveur intercepte et manipule (Totalement chiffré).")
    server_placeholder = st.empty()

# Fonction pour rafraîchir les logs serveur en temps réel
def refresh_server_logs():
    with server_placeholder.container(height=500):
        for log in st.session_state.server_logs:
            st.text(log)

# Rendu initial des logs existants
refresh_server_logs()

# --- LOGIQUE D'EXÉCUTION LORS DE L'ENVOI ---
if prompt:
    # 1. Affichage immédiat du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with col_user:
        with st.chat_message("user"):
            st.markdown(prompt)

    # =============================================
    # MÉTHODES FHE RÉELLES (COMPACT / MOAI)
    # =============================================
    if choix_methode in ["Compact (PoPETS 2024)", "MOAI"]:
        fhe_engine = engines[choix_methode]
        
        with col_user:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                total_fhe_time = 0.0
                
                # Log initial côté serveur
                st.session_state.server_logs.append(
                    f"> [{choix_methode}] Requête reçue. Chiffrement (FHE Rotation-Free).\n"
                    f"> Le serveur ne peut PAS lire la requête ni voir la réponse."
                )
                refresh_server_logs()

                # Génération Token par Token avec le moteur SEAL adéquat
                t_start_gen = time.time()
                token_count = 0
                for step_data in fhe_engine.chat_stream(
                    prompt, 
                    max_tokens=max_tokens, 
                    temperature=temperature, 
                    top_k=top_k, 
                    repetition_penalty=penalty
                ):
                    token_count += 1
                    # --- Vue Client : affichage progressif du texte ---
                    full_response += step_data["word"]
                    message_placeholder.markdown(full_response + "▌")
                    
                    # --- Vue Serveur : log de chaque opération aveugle ---
                    total_fhe_time += step_data["exec_time"]
                    st.session_state.server_logs.append(
                        f"⚙️ [OPÉRATION AVEUGLE] {step_data['enc_bytes_len']} octets chiffrés traités | "
                        f"Temps FHE: {step_data['exec_time']*1000:.1f}ms"
                    )
                    refresh_server_logs()

                # Fin de la génération
                total_duration = time.time() - t_start_gen
                tokens_per_sec = token_count / total_duration if total_duration > 0 else 0
                
                message_placeholder.markdown(full_response)
                st.session_state.server_logs.append(
                    f"✅ [TERMINÉ] Inférence complète.\n"
                    f"⏱️ Temps total : {total_duration:.2f}s | 🚀 Vitesse : {tokens_per_sec:.2f} tok/s\n"
                    f"🔐 Temps FHE cumulé: {total_fhe_time:.2f}s"
                )
                refresh_server_logs()
                st.session_state.messages.append({"role": "assistant", "content": full_response})