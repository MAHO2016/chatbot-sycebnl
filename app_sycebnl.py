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

# === CSS PERSONNALISE ===
st.markdown("""
<style>
    /* Header */
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
    /* Badges */
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
    /* Messages */
    .stChatMessage {
        font-family: Arial, sans-serif;
    }
    /* Footer */
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
    /* Masquer le menu Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("""
<div class="expertasso-header">
    <div class="expertasso-logo">EA</div>
    <div>
        <p class="expertasso-title">ExpertAsso</p>
        <p class="expertasso-subtitle">Assistant IA spécialisé SYCEBNL · Bénin</p>
    </div>
</div>
""", unsafe_allow_html=True)

# === BADGES ===
st.markdown("""
<div class="badge-container">
    <span class="badge badge-blue">📘 Comptabilité SYCEBNL</span>
    <span class="badge badge-gold">⚖️ Législation Associations Bénin</span>
    <span class="badge badge-red">🏛️ Fiscalité CGI Bénin 2026</span>
</div>
""", unsafe_allow_html=True)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

FICHIERS_DRIVE = {
    "audcif.pdf"                      : "1zX4w8Qwg5lLjFkLrMa9LIwGd8lRyly28",
    "guide_application_sycebnl.pdf"   : "1bJuHOBaBYTdvEDispwpjhrVui9hJErSg",
    "guide_application_syscohada.pdf" : "1jktfwwqKxGALOegiMQAZPueKVDYaK177",
    "Livre_sur_les_associations.pdf"  : "1YHrNgCYmacVXUfY0fiWaXiv1G5WGqu3n",
    "loi_sur_les_associations.pdf"    : "1ppwH7BHwiocHBTRN_hUGXiSbFVkcWo-i",
    "sycebnl.pdf"                     : "1O6le6GKQy4AG5fEef5JB9XJX-QX6l39Z",
    "cgi_2026.pdf"                    : "1uoBDsVF-4hftsywou6YEH60yf7Myst26",
    "AU_SYCEBNL.pdf"                  : "1PjwvGKxN4QYA4ke8gZWRYDwSWAcpfI4J",
}

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
    telecharger_fichiers()
    dossier_docs = "documents"
    dossier_db   = "faiss_index"
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


PROMPT = """Tu es un assistant specialise UNIQUEMENT dans :
1. Le SYCEBNL (Systeme Comptable des Entites a But Non Lucratif)
2. La loi sur les associations au Benin
3. Le Code General des Impots (CGI) du Benin 2026

REGLES ABSOLUES :
- Tu utilises UNIQUEMENT les informations contenues dans le contexte fourni
- Tu n'utilises JAMAIS tes connaissances generales
- Si l'information n'est pas dans le contexte, tu reponds :
  "Je n'ai pas de réponse à cette question actuellement."
- Tu ne confonds JAMAIS le SYCEBNL avec le Syscohada
- Tu n'inventes JAMAIS de comptes ou d'articles

REGLES DE COMPLIMENT
- Il faut toujours vouvoyer les interlocuteurs et non les tutoyer.
- Quand quelqu'un te dire Merci, réponds lui poliment, je vous en prie

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

COMPTES DE REFERENCE OBLIGATOIRES :
- Caisse : 5711
- Banque : 5211
- Virement de fonds : 585
- Adherents / Fideles : 4111
- Dimes recues : 7044
- Dons recus : 7041
- Cotisations : 701
- Legs : 7042
- Quetes et assimilees : 7044

ECRITURES TYPES VALIDEES :

E01 — Collecte de la dime / quete / offrande en especes :
  Comptabilite de tresorerie (1 ecriture) :
    Debit  5711 Caisse              X FCFA
    Credit 7044 Dimes/quetes        X FCFA
    Libelle : S/Encaissement des dimes

  Comptabilite d'engagement (2 ecritures) :
    Ecriture 1 :
      Debit  4111 Fideles (Adherents)   X FCFA
      Credit 7044 Dimes/quetes          X FCFA
      Libelle : S/Engagement
    Ecriture 2 :
      Debit  5711 Caisse                X FCFA
      Credit 4111 Fideles (Adherents)   X FCFA
      Libelle : S/Encaissement des quetes

E02 — Versement des especes en banque :
  Tresorerie ET engagement (2 ecritures identiques) :
    Ecriture 1 :
      Debit  5211 Banque               X FCFA
      Credit 585  Virement de fonds    X FCFA
      Libelle : S/Bordereau de versement
    Ecriture 2 :
      Debit  585  Virement de fonds    X FCFA
      Credit 5711 Caisse               X FCFA
      Libelle : S/Avis de credit

E03 — Reception d'un don en especes :
  Comptabilite de tresorerie (1 ecriture) :
    Debit  5711 Caisse    X FCFA
    Credit 7041 Dons      X FCFA
    Libelle : S/Reception don en especes

  Comptabilite d'engagement (2 ecritures) :
    Ecriture 1 :
      Debit  4111 Fideles (Adherents)   X FCFA
      Credit 7041 Dons                  X FCFA
      Libelle : S/Engagement don
    Ecriture 2 :
      Debit  5711 Caisse                X FCFA
      Credit 4111 Fideles (Adherents)   X FCFA
      Libelle : S/Encaissement don

E04 — Reception d'un don par virement bancaire :
  Comptabilite de tresorerie (1 ecriture) :
    Debit  5211 Banque    X FCFA
    Credit 7041 Dons      X FCFA
    Libelle : S/Reception don par virement

  Comptabilite d'engagement (2 ecritures) :
    Ecriture 1 :
      Debit  4111 Fideles (Adherents)   X FCFA
      Credit 7041 Dons                  X FCFA
      Libelle : S/Engagement don
    Ecriture 2 :
      Debit  5211 Banque                X FCFA
      Credit 4111 Fideles (Adherents)   X FCFA
      Libelle : S/Encaissement don

E05 — Reception cotisation en especes :
  Comptabilite de tresorerie (1 ecriture) :
    Debit  5711 Caisse                     X FCFA
    Credit 701  Cotisations des adherents  X FCFA
    Libelle : S/Reception cotisation

  Comptabilite d'engagement (2 ecritures) :
    Ecriture 1 :
      Debit  4111 Fideles (Adherents)        X FCFA
      Credit 701  Cotisations des adherents  X FCFA
      Libelle : S/Engagement cotisation
    Ecriture 2 :
      Debit  5711 Caisse                     X FCFA
      Credit 4111 Fideles (Adherents)        X FCFA
      Libelle : S/Encaissement cotisation

REGLE ABSOLUE SUR LES ECRITURES :
- Remplace X par le montant exact mentionne par l'utilisateur
- Ne changes JAMAIS les numeros de comptes
- Si l'operation ne correspond a aucune ecriture ci-dessus, dis :
  "Cette operation n'est pas encore dans ma base d'ecritures.
   Veuillez contacter un expert-comptable."

ARTICLE FONDAMENTAL — CGI BENIN ARTICLE 4-9 :
Les associations et organismes sans but lucratif legalement constitues et dont
la gestion est desinteressee sont EXONERES de l'impot sur les societes.
Conditions :
a) Gere a titre benevole — remuneration possible si ne depasse pas 10 fois le SMIG.
b) Depot rapport d'activite au plus tard le 30 avril de chaque annee.

DISTINCTIONS OBLIGATOIRES :
1. LIVRES COMPTABLES OBLIGATOIRES selon SYCEBNL (exactement 4) :
   - Le Journal
   - Le Grand Livre
   - La Balance generale des comptes
   - Le Livre d'inventaire

2. DOCUMENTS OBLIGATOIRES mais PAS des livres comptables :
   - Le Registre des donateurs

Que comporte le registre des donateurs?
Le registre des donateurs comporte:
- la date de l’opération ;
- les nom et prénoms, le domicile et l’adresse mail des personnes physiques ;
- la dénomination de la personne morale, 
- le registre de commerce, 
- le numéro d’identification fiscal, 
- l’adresse du siège social et du mail ;
- le montant du don/legs et le mode de libération (espèces, chèques, virement, nature). 
Toutes les écritures contenues dans ce registre doivent être signées par le représentant légal de l’entité à but non lucratif.
Ce registre peut être tenu en version physique reliée, brochée ou en version électronique. 

3. Quelles sont les entités qui sont éligibles au système minimal de trésorerie (SMT)
- Sont éligibles au Système minimal de trésorerie, les entités dont les ressources annuelles sont inférieures ou égales aux seuils suivants 30 millions de FCFA ou l'unité ayant cours légal dans l'Etat partie

4. ETATS FINANCIERS OBLIGATOIRES SELON LE SYCEBNL :

Il existe 3 types d'entites avec des etats financiers differents :

=== 1. ASSOCIATIONS ET ORDRES PROFESSIONNELS ===
Deux systemes possibles :

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
UN SEUL systeme possible : le Systeme Normal (SN) uniquement.
6 etats financiers obligatoires :
   - Tableau emplois-ressources
   - Tableau d'execution budgetaire
   - Tableau de reconciliation de tresorerie
   - Bilan
   - Compte d'exploitation
   - Notes annexes

REGLE ABSOLUE PROJETS DE DEVELOPPEMENT :
Le SMT N'EXISTE PAS pour les projets de developpement.
Si on demande le SMT pour les projets de developpement, repondre :
"Le Systeme Minimal de Tresorerie (SMT) n'existe pas pour les projets
de developpement. Seul le Systeme Normal (SN) est applicable."

=== TABLEAU RECAPITULATIF ===
| Entite                        | SN disponible | SMT disponible |
| Associations et ordres prof.  | OUI           | OUI            |
| Projets de developpement      | OUI           | NON            |

5. CONDITION DE NOMINATION D'UN AUDITEUR
Selon l’article 19 de l’acte uniforme relatif au système comptable des entités à but non lucratif, sont tenues de désigner au moins un (1) auditeur, les entités à but non lucratif qui remplissent, à la clôture de l’exercice, l’un des trois critères suivants:
        - total du bilan supérieur à cent millions (100.000.000) de francs CFA ou l’équivalent dans l'unité monétaire ayant cours légal dans l’Etat partie ;
        - ressources annuelles supérieures à deux cents millions (200.000.000) de francs CFA ou l’équivalent dans l'unité monétaire ayant cours légal dans l’Etat partie ;
        - effectif permanent supérieur à vingt (20) personnes ;
L’entité n’est plus tenue de désigner un auditeur dès lors qu’elle ne remplit plus aucun des trois (3) critères fixés ci-dessus pendant les deux (2) exercices précédent l’expiration du mandat de l’auditeur.
Pour les autres entités à but non lucratif ne remplissant pas ces critères, la nomination de l’auditeur est facultative. Elle peut toutefois être demandée en justice par au moins dix pour cent (10%) des membres.

6. QUI PEUT ETRE NOMME AUDITEUR?
L’auditeur est choisi parmi les experts-comptables inscrits au tableau de l’ordre des experts-comptables ou de l’organe qui en tient lieu, et 

7. QUELLE EST LA DUREE DU MANDAT DE L'AUDITEUR?
L'auditeur est nommé pour trois (3) exercices renouvelables une fois par l’assemblée générale ou l’instance qui en tient lieu de l’entité représentant plus de la moitié des membres présents ou représentés, ou par le bailleur de fonds et/ou l’Etat bénéficiaire du Projet de développement. 
Toutefois, si l’entité à une existence inférieure à trois exercices, le mandat de l’auditeur est ramené à cette durée.



REGLE DE RECHERCHE PAR ARTICLE :
Quand l'utilisateur demande un article specifique du SYCEBNL ou de la loi,
cherche dans le contexte le contenu de cet article.
Si tu ne trouves pas l'article exact, dis :
"Je n'ai pas pu extraire cet article directement. Voici ce que je sais
sur ce sujet d'apres les documents disponibles :"
et donne les informations disponibles sur le sujet.
Ne reponds JAMAIS uniquement "Je ne trouve pas" sans donner d'information utile.
Contexte extrait des documents officiels :
{context}

Historique de la conversation :
{historique}

Question actuelle : {question}

Reponse concise et adaptee :"""


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


# === INTERFACE CHAT ===
llm, retriever = charger_modele()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === LIMITE DE MESSAGES ===
MAX_MESSAGES = 20

if "nombre_questions" not in st.session_state:
    st.session_state.nombre_questions = 0

# Afficher compteur
questions_restantes = MAX_MESSAGES - st.session_state.nombre_questions
if questions_restantes <= 5:
    st.warning(f"⚠️ Il vous reste {questions_restantes} question(s) dans cette session.")

# Mots-cles hors sujet
HORS_SUJET = [
    "recette", "cuisine", "football", "sport", "politique",
    "météo", "film", "musique", "amour", "jeu", "blague",
    "python", "programmation", "code", "intelligence artificielle"
]

def est_hors_sujet(question):
    question_lower = question.lower()
    return any(mot in question_lower for mot in HORS_SUJET)

if question := st.chat_input("Posez votre question sur le SYCEBNL..."):

    # Vérifier limite
    if st.session_state.nombre_questions >= MAX_MESSAGES:
        st.error("Vous avez atteint la limite de 20 questions par session. Cliquez sur 'Nouvelle conversation' pour recommencer.")

    # Vérifier hors sujet
    elif est_hors_sujet(question):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown("Je suis spécialisé uniquement dans le SYCEBNL, la loi sur les associations et la fiscalité au Bénin. Je ne peux pas répondre à cette question.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Je suis spécialisé uniquement dans le SYCEBNL, la loi sur les associations et la fiscalité au Bénin. Je ne peux pas répondre à cette question."
        })

    # Question valide
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                reponse = generer_reponse(
                    llm, retriever,
                    st.session_state.messages,
                    question
                )
            st.markdown(reponse)
        st.session_state.messages.append(
            {"role": "assistant", "content": reponse})
        st.session_state.nombre_questions += 1

# === BOUTON REINITIALISATION ===
col1, col2, col3 = st.columns([4, 2, 4])
with col2:
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()

# === FOOTER ===
st.markdown("""
<div class="expertasso-footer">
    <span class="footer-left">© 2026 ExpertAsso · Bénin</span>
    <span class="footer-right">ComptaProgresso</span>
</div>
""", unsafe_allow_html=True)
