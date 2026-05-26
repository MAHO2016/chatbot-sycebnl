import os
import requests
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="ExpertAsso — Assistant SYCEBNL",
    page_icon="🏛️",
    layout="wide"
)

st.markdown("""
<style>
    .expertasso-header {
        background: #1A3C8A;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 1rem;
    }
    .expertasso-logo {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: #CBA135;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: bold;
        color: #1A3C8A;
        font-family: 'Rockwell', serif;
        flex-shrink: 0;
    }
    .expertasso-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: bold;
        font-family: 'Rockwell', serif;
        margin: 0;
    }
    .expertasso-subtitle {
        color: #CBA135;
        font-size: 13px;
        font-family: Arial, sans-serif;
        margin: 0;
    }
    .badge-container {
        display: flex;
        gap: 10px;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-family: Arial, sans-serif;
        font-weight: 500;
    }
    .badge-blue {
        background: rgba(26,60,138,0.1);
        color: #1A3C8A;
        border: 1px solid #1A3C8A;
    }
    .badge-gold {
        background: rgba(203,161,53,0.1);
        color: #CBA135;
        border: 1px solid #CBA135;
    }
    .badge-red {
        background: rgba(255,0,0,0.08);
        color: #CC0000;
        border: 1px solid #FF0000;
    }
    .expertasso-footer {
        background: #1A3C8A;
        padding: 10px 2rem;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1rem;
    }
    .footer-left {
        color: rgba(255,255,255,0.5);
        font-size: 11px;
        font-family: Arial, sans-serif;
    }
    .footer-right {
        color: #CBA135;
        font-size: 11px;
        font-family: Arial, sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="expertasso-header">
    <div class="expertasso-logo">EA</div>
    <div>
        <p class="expertasso-title">ExpertAsso</p>
        <p class="expertasso-subtitle">Assistant IA specialise SYCEBNL · Benin</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badge-container">
    <span class="badge badge-blue">Comptabilite SYCEBNL</span>
    <span class="badge badge-gold">Legislation Associations Benin</span>
    <span class="badge badge-red">Fiscalite CGI Benin 2026</span>
</div>
""", unsafe_allow_html=True)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

FICHIERS_DRIVE = {
    "audcif.pdf"                     : "1zX4w8Qwg5lLjFkLrMa9LIwGd8lRyly28",
    "guide_application_sycebnl.pdf"  : "1bJuHOBaBYTdvEDispwpjhrVui9hJErSg",
    "guide_application_syscohada.pdf": "1jktfwwqKxGALOegiMQAZPueKVDYaK177",
    "Livre_sur_les_associations.pdf" : "1YHrNgCYmacVXUfY0fiWaXiv1G5WGqu3n",
    "loi_sur_les_associations.pdf"   : "1ppwH7BHwiocHBTRN_hUGXiSbFVkcWo-i",
    "sycebnl.pdf"                    : "1O6le6GKQy4AG5fEef5JB9XJX-QX6l39Z",
    "cgi_2026.pdf"                   : "1uoBDsVF-4hftsywou6YEH60yf7Myst26",
    "AU_SYCEBNL.pdf"                 : "1PjwvGKxN4QYA4ke8gZWRYDwSWAcpfI4J",
}

# === REPONSES SENTIMENTS ===
SENTIMENTS = {
    "merci": "Avec plaisir ! Je suis toujours disponible pour vous aider sur le SYCEBNL et la legislation des associations au Benin. N'hesitez pas si vous avez d'autres questions.",
    "merci beaucoup": "C'est avec grand plaisir ! Votre confiance m'honore. Je reste a votre disposition pour toute question sur le SYCEBNL, la loi sur les associations ou la fiscalite au Benin.",
    "bonjour": "Bonjour ! Je suis ExpertAsso, votre assistant specialise dans le SYCEBNL et la legislation des associations au Benin. Comment puis-je vous aider aujourd'hui ?",
    "bonsoir": "Bonsoir ! Je suis ExpertAsso, votre assistant specialise dans le SYCEBNL et la legislation des associations au Benin. Comment puis-je vous aider ce soir ?",
    "bonne nuit": "Bonne nuit ! N'hesitez pas a revenir demain pour toute question sur le SYCEBNL ou la legislation des associations.",
    "bravo": "Merci pour vos encouragements ! Je fais de mon mieux pour vous fournir des informations precises sur le SYCEBNL et la legislation des associations au Benin.",
    "excellent": "Merci ! Je suis la pour vous accompagner dans votre comprehension du SYCEBNL et de la legislation des associations.",
    "parfait": "Merci ! Si vous avez d'autres questions sur le SYCEBNL ou la legislation des associations, je suis a votre service.",
    "super": "Merci pour votre retour positif ! Je reste a votre disposition pour toute autre question.",
    "bien": "Merci ! N'hesitez pas si vous avez d'autres questions sur le SYCEBNL ou la fiscalite des associations.",
    "ok": "Tres bien ! Y a-t-il autre chose que je puisse faire pour vous concernant le SYCEBNL ou la legislation des associations ?",
    "d'accord": "Parfait ! Je reste disponible pour toute autre question sur le SYCEBNL ou la legislation des associations au Benin.",
    "au revoir": "Au revoir ! Ce fut un plaisir de vous assister. N'hesitez pas a revenir pour toute question sur le SYCEBNL ou la legislation des associations au Benin.",
    "bonne journee": "Merci, bonne journee a vous egalement ! Je reste disponible pour toute question sur le SYCEBNL.",
    "qui es-tu": "Je suis ExpertAsso, un assistant IA specialise dans le SYCEBNL (Systeme Comptable des Entites a But Non Lucratif), la loi sur les associations au Benin et la fiscalite (CGI Benin 2026). Je suis developpe par ComptaProgresso pour accompagner les associations dans leur gestion comptable et administrative.",
    "que peux-tu faire": "Je peux vous aider sur : les obligations comptables des associations (SYCEBNL), la legislation des associations au Benin, les questions fiscales (CGI Benin 2026), les livres comptables obligatoires, les ecritures comptables et bien plus encore !",
}

def detecter_sentiment(question):
    question_lower = question.lower().strip()
    for mot_cle, reponse in SENTIMENTS.items():
        if mot_cle in question_lower:
            return reponse
    return None


def telecharger_fichiers():
    os.makedirs("documents", exist_ok=True)
    for nom, file_id in FICHIERS_DRIVE.items():
        chemin = os.path.join("documents", nom)
        if not os.path.exists(chemin):
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(url, stream=True)
            with open(chemin, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)


@st.cache_resource
def charger_modele():
    import shutil
    telecharger_fichiers()
    dossier_docs = "documents"
    dossier_db = "faiss_index"
    if os.path.exists(dossier_db):
        shutil.rmtree(dossier_db)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    if False:
        vectorstore = FAISS.load_local(
            dossier_db, embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        documents = []
        for fichier in os.listdir(dossier_docs):
            if fichier.endswith('.pdf'):
                chemin = os.path.join(dossier_docs, fichier)
                loader = PyPDFLoader(chemin)
                documents.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        morceaux = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(
            documents=morceaux, embedding=embeddings)
        vectorstore.save_local(dossier_db)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.2,
        api_key=OPENAI_API_KEY)
    return llm, retriever


PROMPT = """Tu es ExpertAsso, un assistant specialise et bienveillant dans :
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

REGLE SUR LES ARTICLES :
- Quand tu trouves le contenu d'un article dans le contexte, cite-le directement
- Ne dis JAMAIS "Je n'ai pas pu extraire cet article directement"
- Si tu trouves l'article, donne son contenu sans introduction inutile

REGLE DE CONCISION :
- Si la question est simple, tu donnes une reponse courte et directe
- Tu ne developpes les details que si la personne les demande explicitement

REGLE POUR LES QUESTIONS SUR LES ECRITURES COMPTABLES :
Si la question porte sur une ecriture comptable et que le type de comptabilite
n'est pas precise dans l'historique, tu demandes TOUJOURS :
"Pour vous donner l'ecriture correcte, pourriez-vous me preciser si votre
association tient une comptabilite d'engagement ou une comptabilite de
tresorerie ?"
Si le type de comptabilite est deja precise dans l'historique, tu donnes
directement l'ecriture sans redemander.

ETATS FINANCIERS OBLIGATOIRES SELON LE SYCEBNL :

=== 1. ASSOCIATIONS ET ORDRES PROFESSIONNELS ===
SYSTEME NORMAL (SN) — 4 etats financiers :
   - Bilan
   - Compte de resultat
   - Tableau des flux de tresorerie
   - Notes annexes

SYSTEME MINIMAL DE TRESORERIE (SMT) — 3 etats financiers :
   - Bilan
   - Compte de resultat
   - Notes annexes
   ATTENTION : Le SMT n'a PAS de Tableau des flux de tresorerie

=== 2. PROJETS DE DEVELOPPEMENT ===
UN SEUL systeme : le Systeme Normal (SN) uniquement.
6 etats financiers :
   - Tableau emplois-ressources
   - Tableau d'execution budgetaire
   - Tableau de reconciliation de tresorerie
   - Bilan
   - Compte d'exploitation
   - Notes annexes

REGLE ABSOLUE : Le SMT N'EXISTE PAS pour les projets de developpement.

ARTICLE FONDAMENTAL — CGI BENIN ARTICLE 4-9 :
Les associations et organismes sans but lucratif legalement constitues et dont
la gestion est desinteressee sont EXONERES de l'impot sur les societes.
Conditions :
a) Gere a titre benevole — remuneration possible si ne depasse pas 10 fois le SMIG.
b) Depot rapport d'activite au plus tard le 30 avril de chaque annee.

COMPTES DE REFERENCE OBLIGATOIRES :
- Caisse : 5711 | Banque : 5211 | Virement de fonds : 585
- Adherents / Fideles : 4111 | Dimes/quetes : 7044
- Dons : 7041 | Cotisations : 701 | Legs : 7042

ECRITURES TYPES VALIDEES :

E01 — Collecte dime/quete/offrande en especes :
  Tresorerie : Debit 5711 / Credit 7044 — Libelle : S/Encaissement dimes
  Engagement : Debit 4111/Credit 7044 puis Debit 5711/Credit 4111

E02 — Versement especes en banque :
  Debit 5211/Credit 585 (bordereau) puis Debit 585/Credit 5711 (avis credit)

E03 — Don en especes :
  Tresorerie : Debit 5711 / Credit 7041
  Engagement : Debit 4111/Credit 7041 puis Debit 5711/Credit 4111

E04 — Don par virement :
  Tresorerie : Debit 5211 / Credit 7041
  Engagement : Debit 4111/Credit 7041 puis Debit 5211/Credit 4111

E05 — Cotisation en especes :
  Tresorerie : Debit 5711 / Credit 701
  Engagement : Debit 4111/Credit 701 puis Debit 5711/Credit 4111

DISTINCTIONS OBLIGATOIRES :
1. LIVRES COMPTABLES OBLIGATOIRES (exactement 4) :
   Journal | Grand Livre | Balance generale | Livre d'inventaire
2. DOCUMENTS OBLIGATOIRES mais PAS livres comptables :
   Registre des donateurs

Contexte extrait des documents officiels :
{context}

Historique de la conversation :
{historique}

Question actuelle : {question}

Reponse concise, precise et bienveillante :"""


def generer_reponse(llm, retriever, historique, question):
    contexte = retriever.invoke(question)
    contexte_formate = "\n\n".join(doc.page_content for doc in contexte)
    historique_formate = ""
    for msg in historique[:-1]:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        historique_formate += f"{role}: {msg['content']}\n"
    prompt_final = PROMPT.format(
        context=contexte_formate,
        historique=historique_formate,
        question=question
    )
    reponse = llm.invoke(prompt_final)
    return reponse.content


MAX_MESSAGES = 20

HORS_SUJET = [
    "recette", "cuisine", "football", "sport", "politique",
    "meteo", "film", "musique", "amour", "jeu", "blague",
]


def est_hors_sujet(question):
    question_lower = question.lower()
    return any(mot in question_lower for mot in HORS_SUJET)


llm, retriever = charger_modele()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "nombre_questions" not in st.session_state:
    st.session_state.nombre_questions = 0

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

questions_restantes = MAX_MESSAGES - st.session_state.nombre_questions
if questions_restantes <= 5:
    st.warning(f"Il vous reste {questions_restantes} question(s) dans cette session.")

if question := st.chat_input("Posez votre question sur le SYCEBNL..."):
    if st.session_state.nombre_questions >= MAX_MESSAGES:
        st.error("Limite de 20 questions atteinte. Cliquez sur 'Nouvelle conversation'.")

    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Detecter sentiment
        reponse_sentiment = detecter_sentiment(question)

        if reponse_sentiment:
            reponse = reponse_sentiment

        elif est_hors_sujet(question):
            reponse = "Je suis specialise uniquement dans le SYCEBNL, la loi sur les associations et la fiscalite au Benin. Je ne peux pas repondre a cette question."

        else:
            with st.spinner("Recherche en cours..."):
                reponse = generer_reponse(
                    llm, retriever,
                    st.session_state.messages,
                    question
                )
            st.session_state.nombre_questions += 1

        with st.chat_message("assistant"):
            st.markdown(reponse)

        st.session_state.messages.append({"role": "assistant", "content": reponse})

col1, col2, col3 = st.columns([4, 2, 4])
with col2:
    if st.button("Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.nombre_questions = 0
        st.rerun()

st.markdown("""
<div class="expertasso-footer">
    <span class="footer-left">2026 ExpertAsso Benin</span>
    <span class="footer-right">ComptaProgresso</span>
</div>
""", unsafe_allow_html=True)
