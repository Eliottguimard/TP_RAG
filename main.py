"""
TP : Chatbot basé sur un RAG (Retrieval-Augmented Generation)
Code de départ : complétez les fonctions marquées « À COMPLÉTER ».
"""

import os

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

# Imports dont vous aurez besoin (à décommenter au fur et à mesure) :
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Configuration (fournie) -----------------------------------------------------
# Fournisseur : "openai" ou "mistral". Par défaut, détecté selon la clé présente dans .env.
PROVIDER = os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "mistral")
PROVIDER = PROVIDER.strip().lower()
DEFAULT_MODELS = {
    "openai": ("gpt-6-luna", "text-embedding-3-small"),
    "mistral": ("mistral-small-latest", "mistral-embed"),
}
if PROVIDER not in DEFAULT_MODELS:
    raise ValueError(f"LLM_PROVIDER inconnu : '{PROVIDER}' (attendu : openai ou mistral)")
CHAT_MODEL = os.getenv("CHAT_MODEL", DEFAULT_MODELS[PROVIDER][0])
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODELS[PROVIDER][1])
# Une base par fournisseur : des embeddings de modèles différents ne sont pas compatibles
PERSIST_DIR = os.getenv("CHROMA_DIR", f"chroma_db_{PROVIDER}")

PAGES = [
    "Intelligence_artificielle_générative",
    "Transformeur_génératif_préentraîné",
    "Google_Gemini",
    "Grand_modèle_de_langage",
    "ChatGPT",
    "LLaMA",
    "Réseaux_antagonistes_génératifs",
    "Apprentissage_auto-supervisé",
    "Apprentissage_par_renforcement",
    "DALL-E",
    "Midjourney",
    "Stable_Diffusion"
]

# PROMPT AVEC RESTRICTION STRICTE AU CONTEXTE
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Vous êtes un assistant spécialisé. Répondez à la question en français, "
        "uniquement à partir du contexte fourni ci-dessous. "
        "Si la réponse ne figure pas dans le contexte, dites clairement que vous ne savez pas.\n\n"
        "Contexte :\n{context}"
    ),
    (
        "human",
        "{question}"
    )
])

# PROMPT SANS LA CONSTRAINTE DE RESTRICTION
"""prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Vous êtes un assistant spécialisé. Répondez à la question en français.\n\n"
        "Contexte :\n{context}"
    ),
    (
        "human",
        "{question}"
    )
])"""


# --- Modèles (fournis) : utilisez get_embeddings() et get_llm() dans votre code ---

def get_embeddings():
    """Renvoie le modèle d'embeddings du fournisseur choisi."""
    if PROVIDER == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        from tokenizers import Regex, Tokenizer, models, pre_tokenizers
        # Sans tokenizer fourni, langchain-mistralai tente d'en télécharger un sur Hugging Face
        # (dépôt protégé, échoue derrière un proxy). Il ne sert qu'à grouper les textes par lots :
        # on utilise un tokenizer local « 1 caractère = 1 token », qui surestime volontairement.
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
        # max_retries / wait_time : l'offre gratuite limite le débit, on réessaie en cas d'erreur 429
        return MistralAIEmbeddings(model=EMBEDDING_MODEL, tokenizer=tokenizer,
                                   max_retries=10, wait_time=5)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm():
    """Renvoie le modèle de chat du fournisseur choisi."""
    if PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=CHAT_MODEL, temperature=0, max_retries=5)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=CHAT_MODEL)


def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    URL = "https://fr.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,  # suit les redirections (sinon une page redirigée revient vide)
    }
    headers = {"User-Agent": "RAG_project/0.0.1"}

    response = requests.get(URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


def load_documents():
    """
    Partie 1 — Télécharge toutes les pages de PAGES et renvoie une liste de Document
    (texte + métadonnées "title" et "source").
    """
    documents = []

    for title in PAGES:
        try:
            # 1. Téléchargement du contenu de la page
            content = get_wikipedia_page(title)

            # 2. Vérification si le contenu est valide et non vide
            if content and content.strip():
                # Formate le titre lisible en remplaçant les tirets bas par des espaces
                readable_title = title.replace("_", " ")
                url = f"https://fr.wikipedia.org/wiki/{title}"

                # Création de l'objet Document de LangChain
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": readable_title,
                        "source": url
                    }
                )
                documents.append(doc)
            else:
                print(f"[Avertissement] La page '{title}' est vide ou n'a pas pu être extraite.")

        except requests.RequestException as e:
            # Gestion des erreurs de réseau/requête HTTP
            print(f"[Erreur] Échec du téléchargement pour la page '{title}' : {e}")

    return documents
    

def get_retriever():
    """
    Partie 3 — Charge ou réutilise la base vectorielle ChromaDB sur le disque,
    puis renvoie un retriever configuré pour k = 4.
    """
    # 1. Instancier l'objet Chroma pointant sur le dossier de persistance
    vectorstore = Chroma(
        collection_name="rag_wikipedia",
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR
    )

    # 2. Vérifier si la base contient déjà des documents
    if vectorstore._collection.count() == 0:
        print("[Info] Base vectorielle vide. Téléchargement et découpage des documents...")
        
        # Charger et découper les documents uniquement si la base est vide
        documents = load_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Ajouter les chunks à la base (sauvegarde automatique sur le disque)
        vectorstore.add_documents(chunks)
        print(f"[Info] {vectorstore._collection.count()} chunks enregistrés sur le disque dans '{PERSIST_DIR}'.")
    else:
        print(f"[Info] Base vectorielle existante chargée depuis le disque ({vectorstore._collection.count()} chunks).")

    # 3. Renvoyer le retriever configuré pour k = 4
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def format_docs(docs):
    """Partie 4 — Concatène les chunks récupérés en un seul texte de contexte."""
    formatted_chunks = []
    
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Document sans titre")
        formatted_chunks.append(
            f"--- Document {i} : {title} ---\n{doc.page_content}"
        )
    
    return "\n\n".join(formatted_chunks)


def rag_answer(query: str):
    """
    Partie 5 — Exécute la chaîne RAG complète :
    récupération, génération en streaming, affichage des sources et retour du résultat.
    """
    # a) Récupérer les chunks pertinents
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    # Formater le contexte à partir des chunks
    context = format_docs(docs)

    # b) Construire la chaîne LangChain
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    # c) Afficher la réponse au fil de l'eau (streaming)
    print("Réponse :\n")
    full_response = ""
    for token in chain.stream({"context": context, "question": query}):
        print(token, end="", flush=True)
        full_response += token
    print("\n")

    # d) Extraction et affichage des sources (URLs uniques tout en conservant l'ordre)
    raw_sources = [doc.metadata.get("source") for doc in docs if doc.metadata.get("source")]
    unique_sources = list(dict.fromkeys(raw_sources))

    print("Sources :")
    for source in unique_sources:
        print(f"- {source}")
    print()

    # e) Renvoyer la réponse générée et la liste des documents utilisés
    return full_response, docs


if __name__ == "__main__":
    print(f'Fournisseur : {PROVIDER} (chat : {CHAT_MODEL}, embeddings : {EMBEDDING_MODEL})')
    print('Building retriever...')
    retriever = get_retriever()

    try:
        while True:
            print('-' * 50)
            print('Posez une question :')
            question = input('> ')
            print()
            rag_answer(question)
            print('\n')

    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
