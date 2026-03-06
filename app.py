import streamlit as st
import time
import random

# Importation dynamique des méthodes depuis ton dossier
from methods import methode_aes, methode_fhe

st.set_page_config(layout="wide", page_title="Secure LLM Chat", page_icon="🔒")

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.title("⚙️ Paramètres d'Encryption")
st.sidebar.markdown("Choisissez l'algorithme à utiliser pour protéger les embeddings.")

choix_methode = st.sidebar.selectbox(
    "Méthode cryptographique :",
    ("Simulation AES", "Simulation FHE")
)

# --- ROUTAGE DE LA LOGIQUE ---
# On assigne la fonction correcte selon le choix de l'utilisateur
if choix_methode == "Simulation AES":
    fonction_encryption = methode_aes.encrypt
elif choix_methode == "Simulation FHE":
    fonction_encryption = methode_fhe.encrypt


# --- INITIALISATION DE LA MÉMOIRE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_logs" not in st.session_state:
    st.session_state.server_logs = []

# --- INTERFACE PRINCIPALE ---
st.title("Chatbot Sécurisé")
st.divider()

col_user, col_server = st.columns(2)

with col_user:
    st.subheader("Vue Utilisateur")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    prompt = st.chat_input("Posez votre question...")

with col_server:
    st.subheader("Vue Serveur (Données interceptées)")
    log_container = st.container(height=500)
    with log_container:
        for log in st.session_state.server_logs:
            st.code(log, language="bash")

# --- EXÉCUTION LORS DE L'ENVOI ---
if prompt:
    # Affiche le message de l'utilisateur instantanément
    st.session_state.messages.append({"role": "user", "content": prompt})
    with col_user:
        with st.chat_message("user"):
            st.markdown(prompt)
            
    # 1. Utilisation de la méthode choisie dans la sidebar (Vue Serveur)
    with col_server:
        with st.spinner(f"Cryptographie en cours via {choix_methode}..."):
            # Appel de la fonction dynamique
            encrypted_payload, temps_exec = fonction_encryption(prompt)
            # Affichage du log avec le temps de traitement
            log_text = f"> REÇU (Temps de chiffrement: {temps_exec}s)\n{encrypted_payload}"
            st.session_state.server_logs.append(log_text)

    # 2. Simulation de la réponse LLM et décryptage (Vue Utilisateur)
    with col_user:
        with st.chat_message("assistant"):
            with st.spinner("Décryptage de la réponse LLM..."):
                time.sleep(1) # Simulation de l'attente réseau
                final_response = f"Réponse sécurisée traitée avec succès en utilisant la méthode : {choix_methode}."
                st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.rerun() # Force le rafraîchissement pour afficher le log avant la réponse