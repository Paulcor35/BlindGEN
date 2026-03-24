import streamlit as st
import time
import sys
import os

# Ajout du path pour trouver le module C++ (.pyd) et le dossier compact_method
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# On importe le vrai moteur FHE C++ depuis compact_method
from compact_method.blind_chat_cpp import BlindChatCpp

st.set_page_config(layout="wide", page_title="BlindGEN - Inférence Souveraine", page_icon="🔒")

# --- MISE EN CACHE DU MOTEUR FHE (chargé une seule fois) ---
@st.cache_resource
def init_fhe_engine():
    return BlindChatCpp()

fhe_engine = init_fhe_engine()

# --- BARRE LATÉRALE : SÉLECTION DE LA MÉTHODE ---
st.sidebar.title("Banc d'essai FHE")
st.sidebar.markdown("Sélectionnez l'architecture papier à évaluer :")

choix_methode = st.sidebar.radio(
    "Architecture d'inférence aveugle :",
    ["Compact (PoPETS 2024)", "MOAI", "HE-SecureNet"]
)

# Paramètres ajustables
max_tokens = st.sidebar.slider("Nombre max de tokens", 5, 200, 50)

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
    log_container = st.container(height=500)
    with log_container:
        for log in st.session_state.server_logs:
            st.code(log, language="bash")

# --- LOGIQUE D'EXÉCUTION LORS DE L'ENVOI ---
if prompt:
    # 1. Affichage immédiat du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with col_user:
        with st.chat_message("user"):
            st.markdown(prompt)

    # =============================================
    # COMPACT (PoPETS 2024) - VRAI MOTEUR C++ SEAL
    # =============================================
    if choix_methode == "Compact (PoPETS 2024)":
        with col_user:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                total_fhe_time = 0.0
                
                # Log initial côté serveur
                st.session_state.server_logs.append(
                    f"> [COMPACT] Requête reçue. Chiffrement Full-FHE (Ciphertext x Ciphertext).\n"
                    f"> Le serveur ne peut PAS lire la requête ni les poids du modèle."
                )

                # Génération Token par Token avec le vrai moteur SEAL
                for step_data in fhe_engine.chat_stream(prompt, max_tokens=max_tokens):
                    # --- Vue Client : affichage progressif du texte ---
                    full_response += step_data["word"]
                    message_placeholder.markdown(full_response + "▌")
                    
                    # --- Vue Serveur : log de chaque opération aveugle ---
                    total_fhe_time += step_data["exec_time"]
                    st.session_state.server_logs.append(
                        f"⚙️ [OPÉRATION AVEUGLE] {step_data['enc_bytes_len']} octets chiffrés traités | "
                        f"Temps FHE: {step_data['exec_time']*1000:.1f}ms"
                    )

                # Fin de la génération
                message_placeholder.markdown(full_response)
                st.session_state.server_logs.append(
                    f"✅ [TERMINÉ] Inférence complète. Temps FHE cumulé: {total_fhe_time:.2f}s"
                )
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    # =============================================
    # MÉTHODES DE SIMULATION (MOAI / HE-SecureNet)
    # =============================================
    elif choix_methode == "MOAI":
        with col_server:
            with st.spinner("Exécution de l'algorithme MOAI..."):
                time.sleep(1.5)
                st.session_state.server_logs.append(
                    f"> REÇU VIA MOAI (Durée: 1.5s)\n"
                    f"> Spécificité : Optimisation du packing (En développement)\n"
                    f"> Ciphertext : [0xA1B2C3_MOAI_SIMULATION...]"
                )
        with col_user:
            with st.chat_message("assistant"):
                reponse = f"**Analyse FHE terminée via MOAI !**\n\nLe serveur a traité vos données à l'aveugle. (Simulation - En développement)"
                st.markdown(reponse)
            st.session_state.messages.append({"role": "assistant", "content": reponse})

    elif choix_methode == "HE-SecureNet":
        with col_server:
            with st.spinner("Exécution de l'algorithme HE-SecureNet..."):
                time.sleep(3)
                st.session_state.server_logs.append(
                    f"> REÇU VIA HE-SECURENET (Durée: 3.2s)\n"
                    f"> Spécificité : Réseau sécurisé spécifique (En développement)\n"
                    f"> Ciphertext : [0x9F8E7D_SECURENET_SIMULATION...]"
                )
        with col_user:
            with st.chat_message("assistant"):
                reponse = f"**Analyse FHE terminée via HE-SecureNet !**\n\nLe serveur a traité vos données à l'aveugle. (Simulation - En développement)"
                st.markdown(reponse)
            st.session_state.messages.append({"role": "assistant", "content": reponse})
