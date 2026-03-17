import streamlit as st
import time
<<<<<<< HEAD
=======
import random
import sys
import os
>>>>>>> c28bd94d0f82c8e0795ab20556fd3c0649cc5534

# --- IMPORTATION DU MOTEUR COMPACT ---
# On importe les classes et la fonction pipeline depuis le dossier engines
from engines.engine_compact import BlindSDK, BlindServer, run_compact_pipeline

<<<<<<< HEAD
st.set_page_config(layout="wide", page_title="BlindGEN Benchmark", page_icon="🔒")

# --- MISE EN CACHE DES MODÈLES LOURDS ---
# @st.cache_resource permet de ne charger les modèles qu'une seule fois au démarrage de l'app
@st.cache_resource
def init_fhe_models():
    with st.spinner("Initialisation du moteur cryptographique FHE (cela peut prendre quelques secondes)..."):
        sdk = BlindSDK()
        server = BlindServer()
    return sdk, server

# Chargement effectif des modèles en mémoire
sdk_compact, server_compact = init_fhe_models()

# --- BARRE LATÉRALE : SÉLECTION DE LA MÉTHODE ---
st.sidebar.title("Banc d'essai FHE")
st.sidebar.markdown("Sélectionnez l'architecture papier à évaluer :")

choix_methode = st.sidebar.radio(
    "Architecture d'inférence aveugle :",
    ["Compact (PoPETS 2024)", "MOAI", "HE-SecureNet"]
)

# --- FONCTION DE ROUTAGE ---
def router_inference(texte, methode):
    """Aiguille la requête vers le bon code selon le choix de l'utilisateur."""
    if methode == "Compact (PoPETS 2024)":
        # Appel du vrai moteur fonctionnel développé par ton collègue
        return run_compact_pipeline(texte, sdk_compact, server_compact)
        
    elif methode == "MOAI":
        # Simulation en attendant que l'équipe termine le code
        time.sleep(1.5)
        return "[0xA1B2C3_MOAI_SIMULATION...]", 1.5, "Optimisation du packing (En développement)"
        
    elif methode == "HE-SecureNet":
        # Simulation en attendant que l'équipe termine le code
        time.sleep(3)
        return "[0x9F8E7D_SECURENET_SIMULATION...]", 3.2, "Réseau sécurisé spécifique (En développement)"

# --- INITIALISATION DE LA MÉMOIRE DU CHAT ---
=======
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
>>>>>>> c28bd94d0f82c8e0795ab20556fd3c0649cc5534
if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_logs" not in st.session_state:
    st.session_state.server_logs = []

# --- INTERFACE PRINCIPALE ---
<<<<<<< HEAD
st.title("Inférence Sécurisée - Benchmark des Architectures")
st.markdown("Démonstration en direct du traitement d'embeddings chiffrés.")
=======
st.title("Chatbot SOUVERAIN (Full-FHE)")
>>>>>>> c28bd94d0f82c8e0795ab20556fd3c0649cc5534
st.divider()

col_user, col_server = st.columns(2)

# --- COLONNE GAUCHE : CLIENT ---
with col_user:
<<<<<<< HEAD
    st.subheader("👤 Vue Client (Local)")
=======
    st.subheader("Vue Utilisateur (Client)")
>>>>>>> c28bd94d0f82c8e0795ab20556fd3c0649cc5534
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    prompt = st.chat_input("Envoyer un texte à analyser en toute sécurité...")

# --- COLONNE DROITE : SERVEUR ---
with col_server:
<<<<<<< HEAD
    st.subheader("Vue Serveur (Cloud)")
    st.info("Ce que le serveur intercepte et manipule (Totalement chiffré).")
    log_container = st.container(height=500)
    with log_container:
        for log in st.session_state.server_logs:
            st.code(log, language="bash")
=======
    st.subheader("Vue Serveur (Données interceptées)")
    log_container = st.empty()
    
    def render_logs():
        with log_container.container(height=500):
            for log in st.session_state.server_logs:
                st.code(log, language="bash")
                
    render_logs()

>>>>>>> c28bd94d0f82c8e0795ab20556fd3c0649cc5534

# --- LOGIQUE D'EXÉCUTION LORS DE L'ENVOI ---
if prompt:
    # 1. Affichage immédiat du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with col_user:
        with st.chat_message("user"):
            st.markdown(prompt)
<<<<<<< HEAD
            
    # 2. Routage vers le moteur FHE sélectionné (Affiché côté Serveur)
    with col_server:
        with st.spinner(f"Exécution de l'algorithme {choix_methode}..."):
            
            # ---> C'est ici que la magie opère <---
            resultat_chiffre, temps_exec, desc = router_inference(prompt, choix_methode)
            
            # Formatage du log pour prouver au jury que la donnée est chiffrée
            log_serveur = f"> REÇU VIA {choix_methode.upper()} (Durée: {temps_exec}s)\n"
            log_serveur += f"> Spécificité : {desc}\n"
            log_serveur += f"> Ciphertext manipulé : {resultat_chiffre}"
            
            st.session_state.server_logs.append(log_serveur)

    # 3. Retour au client, déchiffrement et affichage final
    with col_user:
        with st.chat_message("assistant"):
            with st.spinner("Déchiffrement local (Clé Secrète)..."):
                # Simulation d'un petit temps de réseau retour
                time.sleep(0.5) 
                
                reponse = f"**Analyse FHE terminée via {choix_methode} !**\n\n"
                reponse += f"Le serveur a traité vos données à l'aveugle. "
                reponse += f"Voici les métadonnées de l'opération :\n- **Temps total :** {temps_exec}s\n- **Détails :** {desc}"
                
                st.markdown(reponse)
                
        st.session_state.messages.append({"role": "assistant", "content": reponse})
=======

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
>>>>>>> c28bd94d0f82c8e0795ab20556fd3c0649cc5534
