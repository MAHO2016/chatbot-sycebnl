import os
import requests
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Assistant SYCEBNL", layout="wide")
st.title("Assistant SYCEBNL — Associations au Benin")
st.markdown("Posez vos questions sur le SYCEBNL, la loi sur les associations et la fiscalite au Benin.")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# IDs des fichiers Google Drive
FICHIERS_DRIVE = {
    "audcif.pdf"                      : "1zX4w8Qwg5lLjFkLrMa9LIwGd8lRyly28",
    "guide_application_sycebnl.pdf"   : "1bJuHOBaBYTdvEDispwpjhrVui9hJErSg",
    "guide_application_syscohada.pdf" : "1jktfwwqKxGALOegiMQAZPueKVDYaK177",
    "Livre_sur_les_associations.pdf"  : "1YHrNgCYmacVXUfY0fiWaXiv1G5WGqu3n",
    "loi_sur_les_associations.pdf"    : "1ppwH7BHwiocHBTRN_hUGXiSbFVkcWo-i",
    "sycebnl.pdf"                     : "1O6le6GKQy4AG5fEef5JB9XJX-QX6l39Z",
    "cgi_2026.pdf"                    : "1uoBDsVF-4hftsywou6YEH60yf7Myst26",
}

def telecharger_fichiers():
    os.makedirs("documents", exist_ok=True)
    for nom, file_id in FICHIERS_DRIVE.items():
        chemin = os.path.join("documents", nom)
        if not os.path.exists(chemin):
            st.write(f"Telechargement : {nom}...")
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(url, stream=True)
            with open(chemin, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    st.write("Tous les documents sont prets !")

@st.cache_resource
def charger_chaine():
    telecharger_fichiers()

    dossier_docs = "documents"
    dossier_db   = "chroma_db"

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    if os.path.exists(dossier_db) and len(os.listdir(dossier_db)) > 0:
        vectorstore = Chroma(
            persist_directory=dossier_db,
            embedding_function=embeddings
        )
    else:
        st.write("Indexation des documents en cours... (5-10 minutes)")
        documents = []
        for fichier in os.listdir(dossier_docs):
            if fichier.endswith('.pdf'):
                chemin = os.path.join(dossier_docs, fichier)
                st.write(f"Chargement : {fichier}")
                loader = PyPDFLoader(chemin)
                documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        morceaux = splitter.split_documents(documents)
        st.write(f"Total morceaux : {len(morceaux)}")

        vectorstore = Chroma.from_documents(
            documents=morceaux,
            embedding=embeddings,
            persist_directory=dossier_db
        )
        st.write("Base vectorielle creee !")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt_template = PromptTemplate.from_template("""
Tu es un assistant specialise UNIQUEMENT dans :
1. Le SYCEBNL (Systeme Comptable des Entites a But Non Lucratif)
2. La loi sur les associations au Benin
3. Le Code General des Impots (CGI) du Benin 2026

REGLES ABSOLUES :
- Tu utilises UNIQUEMENT les informations contenues dans le contexte fourni
- Tu n'utilises JAMAIS tes connaissances generales
- Si l'information n'est pas dans le contexte, tu reponds :
  "Je ne trouve pas cette information dans les documents disponibles."
- Tu ne confonds JAMAIS le SYCEBNL avec le Syscohada
- Tu n'inventes JAMAIS de comptes ou d'articles

REGLE DE CONCISION :
- Si la question est simple, tu donnes une reponse courte et directe
- Tu ne developpes les details que si la personne les demande explicitement

REGLE POUR LES QUESTIONS SUR LES ECRITURES COMPTABLES :
Si la question porte sur une ecriture comptable, tu reponds UNIQUEMENT :
"Pour vous donner l'ecriture correcte, pourriez-vous me preciser si votre
association tient une comptabilite d'engagement ou une comptabilite de
tresorerie ?"
Tu n'essaies JAMAIS de donner une ecriture sans connaitre le type de
comptabilite de l'association.

ARTICLE FONDAMENTAL — CGI BENIN ARTICLE 4-9 :
Les associations et organismes sans but lucratif legalement constitues et dont
la gestion est desinteressee sont EXONERES de l'impot sur les societes.
Conditions de la gestion desinteressee :
a) Gere et administre a titre benevole — remuneration possible si transparence
   financiere et remuneration mensuelle ne depassant pas 10 fois le SMIG.
b) Obligation de deposer le rapport d'activite au plus tard le 30 avril
   de chaque annee aupres des services fiscaux.
Les systemes financiers decentralises sous forme associative ne sont exoneres
que pour la collecte d'epargne et la distribution de credit.

DISTINCTIONS OBLIGATOIRES :
1. LIVRES COMPTABLES OBLIGATOIRES selon SYCEBNL (exactement 4) :
   - Le Journal
   - Le Grand Livre
   - La Balance generale des comptes
   - Le Livre d'inventaire

2. DOCUMENTS OBLIGATOIRES mais PAS des livres comptables :
   - Le Registre des donateurs

Contexte extrait des documents officiels :
{context}

Question : {question}

Reponse concise et adaptee au niveau de la question posee :
""")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=OPENAI_API_KEY
    )

    chaine = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chaine

# Interface chat
chaine = charger_chaine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Posez votre question sur le SYCEBNL..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            reponse = chaine.invoke(question)
        st.markdown(reponse)

    st.session_state.messages.append({"role": "assistant", "content": reponse})
