import streamlit as st
import time
import sys
import os
import torch
import base64
import random
import string
import textwrap
from huggingface_hub import scan_cache_dir

# Ajout du path pour trouver les modules C++ (.pyd) et les dossiers méthodes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# On importe les moteurs FHE
from compact_method.blind_chat_cpp import BlindChatCpp
from moai_method.blind_chat_moai import BlindChatMoai

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

st.set_page_config(layout="wide", page_title="BlindGEN - Inférence Souveraine", page_icon="🔒")

# --- INITIALISATION DE L'ÉTAT ---
if "generating" not in st.session_state:
    st.session_state.generating = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_logs" not in st.session_state:
    st.session_state.server_logs = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "current_fhe_time" not in st.session_state:
    st.session_state.current_fhe_time = 0.0
if "encrypted_buffer" not in st.session_state:
    st.session_state.encrypted_buffer = ""

# Persistance des dernières métriques
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {"tps": 0.0, "fhe_t": 0.0, "p_kb": 0.0, "total_t": 0.0}

# --- HELPERS ---
def wrap_b64(b64_str, width=64):
    """Coupe proprement les longues chaines Base64 pour l'affichage."""
    if not b64_str: return ""
    return "\n".join(textwrap.wrap(b64_str, width))

# --- DETECTION AUTOMATIQUE DES LLM LOCAUX ---
def get_local_llms():
    llms = []
    if os.path.isdir("gpt2_local"):
        llms.append("gpt2_local (Local)")
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_type == "model":
                repo_id = repo.repo_id
                try:
                    config = AutoConfig.from_pretrained(repo_id, local_files_only=True)
                    archs = getattr(config, "architectures", [])
                    if any("CausalLM" in a or "GPT" in a or "ForLM" in a or "Llama" in a or "Qwen" in a for a in archs):
                        llms.append(repo_id)
                except Exception: pass
    except Exception: pass
    llms = sorted(list(set(llms)))
    llms.append("--- Saisie Manuelle ---")
    return llms

@st.cache_resource
def load_engine(model_id, method, poly_n, scale_pow):
    path_to_load = "gpt2_local" if model_id == "gpt2_local (Local)" else model_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(path_to_load)
        model = AutoModelForCausalLM.from_pretrained(path_to_load)
        model.eval()
        h_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
        if method == "MOAI":
            engine = BlindChatMoai(tokenizer, model, poly_n=poly_n, scale_bits=scale_pow)
        else:
            engine = BlindChatCpp(tokenizer, model, poly_n=poly_n, scale_bits=scale_pow)
        return engine, h_size
    except Exception as e:
        st.error(f"Erreur d'initialisation : {e}")
        return None, None

# --- BARRE LATÉRALE ---
st.sidebar.title("Banc d'essai FHE")

# 1. Sélection du modèle
model_list = get_local_llms()
choix_model = st.sidebar.selectbox("Sélectionnez le LLM :", model_list, index=0)
if choix_model == "--- Saisie Manuelle ---":
    choix_model = st.sidebar.text_input("Identifiant Hugging Face :")

# 2. Sélection de l'architecture
choix_methode = st.sidebar.radio("Architecture d'inférence aveugle :", ["Compact (PoPETS 2024)", "MOAI"], index=1)

st.sidebar.divider()

# 3. Paramètres FHE (N et Scale)
st.sidebar.subheader("Configuration SEAL")
fhe_profiles = {
    "Vitesse (8192)": (8192, 30),
    "Industriel (16384)": (16384, 40),
    "Sécurité Max (32768)": (32768, 50)
}
profil_selection = st.sidebar.select_slider(
    "Profil de chiffrement (N / Scale) :",
    options=list(fhe_profiles.keys()),
    value="Industriel (16384)",
    help="Définit le compromis entre sécurité et vitesse. Un degré (N) élevé permet des calculs plus profonds mais est beaucoup plus lent."
)
poly_n_val, scale_bits_val = fhe_profiles[profil_selection]

engine, h_size = None, None
if choix_model and choix_model != "--- Saisie Manuelle ---":
    engine, h_size = load_engine(choix_model, choix_methode, poly_n_val, scale_bits_val)

st.sidebar.divider()

# 4. Largeur FHE en pourcentage
if h_size:
    fhe_percent = st.sidebar.slider(
        "Largeur FHE (%) :", 1, 100, 33,
        help="Plus ce pourcentage est faible, plus le calcul est rapide, mais plus le modèle perd en précision (les neurones 'aveugles' sont ignorés)."
    )
    fhe_slice_abs = int((fhe_percent / 100.0) * h_size)
    fhe_slice_abs = max(1, min(fhe_slice_abs, h_size))
    st.sidebar.write(f"Dimension réelle : **{fhe_slice_abs}** / {h_size} neurones")
else:
    fhe_slice_abs = 128

st.sidebar.divider()

# 5. Paramètres de génération
st.sidebar.subheader("Paramètres")
max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=1000, value=25)
temperature = st.sidebar.number_input("Température", min_value=0.01, max_value=5.0, value=0.70, step=0.05)

# --- INTERFACE PRINCIPALE ---
st.title("Inférence Sécurisée - Benchmark des Architectures")
col_user, col_server = st.columns(2)

with col_server:
    st.subheader("Vue Serveur (Cloud)")
    server_placeholder = st.empty()

def refresh_server_view():
    with server_placeholder.container(height=520):
        for log in st.session_state.server_logs:
            st.text(log)
        if st.session_state.encrypted_buffer:
            st.markdown("**Interception Server (Ciphertexts) :**")
            st.code(wrap_b64(st.session_state.encrypted_buffer), language="text")

with col_user:
    st.subheader("Vue Client (Local)")
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Zone d'input
    input_zone = st.container()
    with input_zone:
        if not st.session_state.generating:
            raw_prompt = st.chat_input("Envoyer un texte à analyser en toute sécurité...")
            if raw_prompt:
                st.session_state.pending_prompt = raw_prompt
                st.session_state.generating = True
                st.session_state.current_fhe_time = 0.0
                st.session_state.encrypted_buffer = ""
                st.rerun()
        else:
            st.markdown("---")
            c1, c2 = st.columns([9, 1])
            with c1: st.info("🔄 Inférence FHE en cours...")
            with c2: 
                if st.button("⏹️", use_container_width=True):
                    st.session_state.generating = False
                    st.rerun()
    
    # Zone des métriques persitante
    metrics_placeholder = st.empty()

def render_metrics(tps=None, fhe_t=None, p_kb=None, total_t=None):
    if tps is None: tps = st.session_state.last_metrics["tps"]
    if fhe_t is None: fhe_t = st.session_state.last_metrics["fhe_t"]
    if p_kb is None: p_kb = st.session_state.last_metrics["p_kb"]
    if total_t is None: total_t = st.session_state.last_metrics["total_t"]
    
    st.session_state.last_metrics = {"tps": tps, "fhe_t": fhe_t, "p_kb": p_kb, "total_t": total_t}
    
    with metrics_placeholder.container():
        r1_c1, r1_c2 = st.columns(2)
        r2_c1, r2_c2 = st.columns(2)
        r1_c1.metric("Vitesse", f"{tps:.2f} tok/s")
        r1_c2.metric("Temps FHE", f"{fhe_t:.2f}s")
        r2_c1.metric("Paquet", f"{p_kb:.1f} KB")
        r2_c2.metric("Durée totale", f"{total_t:.2f}s")

render_metrics()
refresh_server_view()

# --- LOGIQUE D'EXÉCUTION ---
if st.session_state.generating and st.session_state.pending_prompt and engine:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            st.session_state.server_logs.append(f"> [{choix_methode}] Requête FHE. Profil: {profil_selection}")
            refresh_server_view()
            
            t_start_gen = time.time()
            token_count = 0
            
            try:
                for step_data in engine.chat_stream(prompt, max_tokens=max_tokens, temperature=temperature, fhe_slice=fhe_slice_abs):
                    token_count += 1
                    full_response += step_data["word"]
                    message_placeholder.markdown(full_response + "▌")
                    
                    st.session_state.current_fhe_time += step_data["exec_time"]
                    
                    real_noise = step_data.get("ciphertext_b64", "")
                    st.session_state.encrypted_buffer = real_noise
                    
                    elapsed = time.time() - t_start_gen
                    current_tps = token_count / elapsed if elapsed > 0 else 0
                    current_pkb = step_data['enc_bytes_len'] / 1024
                    
                    render_metrics(current_tps, st.session_state.current_fhe_time, current_pkb, elapsed)
                    refresh_server_view()

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Erreur : {e}")
            finally:
                st.session_state.generating = False
                st.rerun()