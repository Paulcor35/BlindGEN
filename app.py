import streamlit as st
import time
import random
import sys
import os

# Importation dynamique des méthodes depuis ton dossier
from methods import methode_MOAI, methode_compact, methode_HE_SecureNet

# On importe le vrai moteur FHE C++
from compact_method.blind_chat_cpp import BlindChatCpp

st.set_page_config(layout="wide", page_title="Secure LLM Chat", page_icon="🔒")

# --- INITIALISATION DU MOTEUR C++ FHE (MISE EN CACHE) ---
# On utilise le cache de Streamlit pour ne pas recharger GPT-2 et Microsoft SEAL à chaque interaction
@st.cache_resource
def load_fhe_engine():
    try:
        return BlindChatCpp()
    except Exception as e:
        print(f"Erreur d'initialisation du moteur C++: {e}")
        return None

fhe_engine = load_fhe_engine()

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.title("Paramètres d'Encryption")
st.sidebar.markdown("Choisissez l'algorithme à utiliser pour protéger les données.")

choix_methode = st.sidebar.selectbox(
    "Méthode cryptographique :",
    (
        "Zéro-Connaissance (Microsoft SEAL C++)", 
        "Simulation Compact", 
        "Simulation MOAI", 
        "Simulation HE-SecureNet"
    )
)

# --- INITIALISATION DE LA MÉMOIRE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_logs" not in st.session_state:
    st.session_state.server_logs = []

# --- INTERFACE PRINCIPALE ---
st.title("Chatbot SOUVERAIN (Full-FHE)")
st.divider()

col_user, col_server = st.columns(2)

with col_user:
    st.subheader("Vue Utilisateur (Client)")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    prompt = st.chat_input("Posez votre question...")

with col_server:
    st.subheader("Vue Serveur (Données interceptées)")
    log_container = st.empty()
    
    def render_logs():
        with log_container.container(height=500):
            for log in st.session_state.server_logs:
                st.code(log, language="bash")
                
    render_logs()


# --- EXÉCUTION LORS DE L'ENVOI ---
if prompt:
    # Affiche le message de l'utilisateur instantanément
    st.session_state.messages.append({"role": "user", "content": prompt})
    with col_user:
        with st.chat_message("user"):
            st.markdown(prompt)

    # -------------------------------------------------------------
    # INTÉGRATION DU VRAI MOTEUR C++ SEAL
    # -------------------------------------------------------------
    if choix_methode == "Zéro-Connaissance (Microsoft SEAL C++)":
        if not fhe_engine:
            st.error("Le moteur C++ n'a pas pu être chargé (vérifiez la compilation).")
        else:
            with col_user:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    st.session_state.server_logs.append("> INITIALISATION FHE : Le serveur reçoit les poids chiffrés.")
                    render_logs()
                    
                    # Génération Token par Token
                    for step_data in fhe_engine.chat_stream(prompt, max_tokens=15):
                        # --- Vue Utilisateur ---
                        full_response += step_data["word"]
                        message_placeholder.markdown(full_response + "▌")
                        
                        # --- Vue Serveur ---
                        # Le serveur enregistre qu'il a traité les octets sans voir le mot
                        log_text = f"⚙️ [OPÉRATION AVEUGLE] {step_data['enc_bytes_len']} octets chiffrés reçus. Temps d'exécution FHE: {step_data['exec_time']:.4f}s"
                        st.session_state.server_logs.append(log_text)
                        render_logs()
                        
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # -------------------------------------------------------------
    # MÉTHODES DE SIMULATION ORIGINALES (PYTHON)
    # -------------------------------------------------------------
    else:
        if choix_methode == "Simulation Compact":
            fonction_encryption = methode_compact.encrypt
        elif choix_methode == "Simulation MOAI":
            fonction_encryption = methode_MOAI.encrypt
        elif choix_methode == "Simulation HE-SecureNet":
            fonction_encryption = methode_HE_SecureNet.encrypt
            
        with col_server:
            with st.spinner(f"Cryptographie en cours via {choix_methode}..."):
                encrypted_payload, temps_exec = fonction_encryption(prompt)
                log_text = f"> REÇU (Temps de chiffrement: {temps_exec}s)\n{encrypted_payload}"
                st.session_state.server_logs.append(log_text)
                render_logs()

        with col_user:
            with st.chat_message("assistant"):
                with st.spinner("Décryptage de la réponse LLM..."):
                    time.sleep(1) # Simulation
                    final_response = f"Réponse sécurisée traitée avec succès en utilisant la méthode : {choix_methode}."
                    st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})