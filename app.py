import streamlit as st
from main import get_retriever, rag_answer

import os
import streamlit as st
from dotenv import load_dotenv

# 1. Charger d'abord les variables du fichier .env local (s'il existe)
load_dotenv()

# 2. Récupérer les clés depuis st.secrets si l'application est sur Streamlit Cloud
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if "MISTRAL_API_KEY" in st.secrets:
        os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
except Exception:
    # En local, si secrets.toml n'existe pas, st.secrets lève une exception qu'on ignore
    pass

# Import de main.py APRÈS avoir configuré os.environ
from main import get_retriever, rag_answer

# Configuration de la page Streamlit
st.set_page_config(page_title="Chatbot RAG Wikipédia", page_icon="🤖")
st.title("🤖 Chatbot RAG Wikipédia")

# --- Initialisation du retriever une seule fois grâce au cache Streamlit ---
@st.cache_resource
def load_retriever():
    return get_retriever()

with st.spinner("Chargement de la base vectorielle..."):
    retriever = load_retriever()

# --- Initialisation de l'historique de conversation dans st.session_state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Affichage de l'historique de la conversation ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Affichage des chunks dans un expander si présents dans l'historique
        if "chunks" in message and message["chunks"]:
            with st.expander("🔍 Passages récupérés (Chunks)"):
                for i, doc in enumerate(message["chunks"], 1):
                    title = doc.metadata.get("title", "Document")
                    st.markdown(f"**Chunk {i} — *{title}***")
                    st.caption(doc.page_content)
                    st.divider()

        # Affichage des sources dans un expander si présentes dans l'historique
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources utilisées"):
                for src in message["sources"]:
                    st.markdown(f"- [{src}]({src})")

# --- Zone de saisie utilisateur ---
if prompt_input := st.chat_input("Posez votre question sur le corpus Wikipédia..."):
    # 1. Afficher et enregistrer le message de l'utilisateur
    st.chat_message("user").markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    # 2. Générer la réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération en cours..."):
            response_text, docs = rag_answer(prompt_input)
            
            # Extraction des URLs uniques tout en préservant l'ordre
            raw_sources = [doc.metadata.get("source") for doc in docs if doc.metadata.get("source")]
            unique_sources = list(dict.fromkeys(raw_sources))

            # Affichage de la réponse
            st.markdown(response_text)

            # Affichage des passages (chunks) extraits
            if docs:
                with st.expander("🔍 Passages récupérés (Chunks)"):
                    for i, doc in enumerate(docs, 1):
                        title = doc.metadata.get("title", "Document")
                        st.markdown(f"**Chunk {i} — *{title}***")
                        st.caption(doc.page_content)
                        st.divider()
            
            # Affichage des liens cliquables vers les sources
            if unique_sources:
                with st.expander("📚 Sources utilisées"):
                    for src in unique_sources:
                        st.markdown(f"- [{src}]({src})")

    # 3. Enregistrer la réponse de l'assistant dans l'historique avec ses chunks et ses sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "chunks": docs,
        "sources": unique_sources
    })