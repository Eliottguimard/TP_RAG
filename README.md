# 🤖 Chatbot RAG Wikipédia

Application web de Chatbot basée sur le principe de **Retrieval-Augmented Generation (RAG)**. Le système interroge une base vectorielle construite à partir d'un ensemble d'articles Wikipédia et génère des réponses avec contrainte stricte tout en affichant ses sources et les extraits consultés.

---

## 🚀 Prérequis

* **Python 3.10+**
* Une clé d'API valide pour **OpenAI** ou **Mistral AI**

---

## 🛠️ Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/Eliottguimard/TP_RAG.git
   cd TP_RAG
   ```

2. **Créer et activer un environnement virtuel :**
   * **Windows (PowerShell) :**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * **macOS / Linux :**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔑 Configuration (.env)

1. Créez un fichier `.env` à la racine à partir du modèle :
   ```bash
   cp .env.template .env
   ```
2. Renseignez votre clé d'API dans `.env` :
   ```env
   OPENAI_API_KEY=votre_cle_api_ici
   # ou
   MISTRAL_API_KEY=votre_cle_api_ici
   ```

---

## 💻 Lancement

### Application Web (Streamlit)
```bash
python -m streamlit run app.py
```
Accès navigateur : `http://localhost:8501`

### Mode Terminal
```bash
python main.py
```

---

## 📂 Structure du Projet

* `app.py` : Interface web Streamlit (`st.session_state`, `@st.cache_resource`, `st.expander`).
* `main.py` : Pipeline RAG (ChromaDB, embeddings, prompt contraint LangChain).
* `.env.template` : Modèle de configuration des clés d'API.
* `.gitignore` : Exclusions Git (`.env`, `.venv/`, `chroma_db_*/`, `__pycache__/`).