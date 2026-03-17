import streamlit as st
import time

# --- IMPORTATION DU MOTEUR COMPACT ---
# On importe les classes et la fonction pipeline depuis le dossier engines
from engines.engine_compact import BlindSDK, BlindServer, run_compact_pipeline

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
if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_logs" not in st.session_state:
    st.session_state.server_logs = []

# --- INTERFACE PRINCIPALE ---
st.title("Inférence Sécurisée - Benchmark des Architectures")
st.markdown("Démonstration en direct du traitement d'embeddings chiffrés.")
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
    st.subheader("Vue Serveur (Cloud)")
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