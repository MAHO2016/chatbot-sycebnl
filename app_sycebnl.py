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


ARTICLES OFFICIELS DE L'ACTE UNIFORME SYCEBNL — PRIORITE ABSOLUE :

Article premier
Il est institué un système comptable unique, commun à tous les États parties, dénommé Système comptable des entités à but non lucratif, en abrégé SYCEBNL, annexé au présent Acte uniforme.
Toute entité à but non lucratif au sens de l'article 2 ci-dessous, qui a son siège dans l'un des États parties au Traité relatif à l'harmonisation du droit des affaires en Afrique ou qui exerce ses activités sur le territoire dudit État, est soumise aux dispositions du présent Acte uniforme.

Article 2
L'entité à but non lucratif s'entend de toute organisation, poursuivant un but désintéresse, et dont les ressources éventuellement générées par l'activité servent au fonctionnement et à la réalisation de son objet social.
Constituent, notamment, des entités à but non lucratif :
1) les associations et les ordres professionnels :
2) les entités ayant pour objet la gestion ou l'administration de projets de développement financés en général par les bailleurs bilatéraux, multilatéraux, privés ou étatiques.
Les entités visées ci-dessus, lorsqu'elles ne sont pas soumises au système de la comptabilité publique, au système de comptabilité soumis à un régime particulier, ou à des dispositions nationales spécifiques, sont tenues de mettre en place une comptabilité, dite comptabilité financière, conformément aux règles applicables au Système comptable des entités à but non lucratif et prévues par les dispositions ci-après.

Article 3
Les dispositions de l'Acte uniforme relatif au droit comptable et à l'information financière sont applicables aux entités visées à l'article 2 ci-dessus, à exception des articles 5. X. 10a 13.17 alinéas 7 et 8, 18. 19 quatrième tiret, 21. 25 à 34, 49, 60. 70, 71. 73 à 113.
Chapitre 2 - États financiers annuels

Article 4
Un jeu complet d'états financiers annuels comprend :
•	pour les associations et les ordres professionnels. 
        o	le Bilan. 
        o	le Compte de résultat. 
        o	Le Tableau des flux de trésorerie 
        o	ainsi que les Notes annexes
•	pour les projets de développement :
        o	le Tableau emplois-ressources,
        o	le Tableau d'exécution budgétaire
        o	le Tableau de réconciliation de trésorerie. 
        o	le Bilan. 
        o	le Compte d'exploitation, 
        o	et les Notes annexes.
Les états financiers forment un tout indissociable et décrivent de façon régulière et sincère les événements, opérations et situations de l'exercice pour donner une image fidèle du patrimoine et de la situation financière de l'entité.
Les états financiers sont établis et présentés conformément aux modèles du Système comptable des entités à but non lucratif.
Toute entité qui applique correctement le Système comptable des entités à but non lucratif est réputée donner, dans ses états financiers. Une image fidèle de sa situation et de ses opérations.
Lorsque l'application d'une prescription comptable se révèle insuffisante ou inadaptée pour donner l'image fidèle, des informations complémentaires ou des justifications nécessaires sont obligatoirement fournies dans les Notes annexes.

Article 5
Les états financiers annuels visés à l'article 4 sont rendus obligatoires, en tout ou en partie, en fonction de la taille des entités appréciée selon des critères mentionnés à l'article 6 ci-après.
Les présentations des états financiers annuels et la tenue des comptes admises sont le Système normal et le Système minimal de trésorerie. Selon le cas. Toute entité est, sauf exception lice à sa taille. Soumise au Système normal de présentation des états financiers et de tenue des comptes.
Le régime juridique du dépôt des états financiers est soumis à la législation interne de chaque État partie.

Article 6
Les petites entités sont assujetties, sauf option, au Système minimal de trésorerie en abroge SMT.
Sont éligibles au Système minimal de trésorerie, les entités dont les ressources annuelles son inférieures ou égales aux seuils suivant.
1) subventions : trente millions (30.000.000) de francs CFA ou l'équivalent dans l'unité monétaire ayant cours légal dans l'État partie :
2) cotisations et autres revenus : trente millions (30.000.000) de francs CFA ou l'équivalent dans l'unité monétaire ayant cours légal dans l'Etat partie :
3) dons et/ou legs : trente millions (30.000.000) de francs CFA ou l'équivalent dans l'unité monétaire ayant cours légal dans l'État partie :
4) ressources du projet de développement : trente millions (30.000.000) de francs CFA ou l'équivalent dans l'unité monétaire ayant cours légal dans l'Etat partie :
5) autres ressources : trente millions (30.000.000) de francs CFA ou l'équivalent dans l'unité monétaire ayant cours légal dans l'Etat partie.
Si l'une des ressources citées ci-dessus est supérieure aux seuils susmentionnés ou si, de manière cumulée sur deux exercices, les ressources dépassent trente millions (30.000.000) de francs CFA ou l'équivalent dans l'unité monétaire ayant cours légal dans l'Etat partie. L’entité est éligible au Système normal.

Article 7
Le Bilan décrit séparément les éléments d'actif et les éléments de passif constituant le patrimoine de l'entité. Il fait apparaître de façon distincte les fonds propres de l'entité.
Le Compte de résultat des associations et ordres professionnels récapitule en liste, les produits et les charges qui font apparaître l'excédent net ou le déficit net de l'exercice. Il doit être procédé, dans l'exercice à tous amortissements, dépréciations et provisions nécessaires pour tenir compte des consommations d'avantages économiques, des pertes de valeurs, des risques et des charges probables, même en cas d'absence ou d'insuffisance d'excédent.
Il doit être tenu compte des risques, charges et produits intervenus au cours de l'exercice ou d'un exercice antérieur, même s'ils sont connus seulement entre la date de clôture de l'exercice et celle de l'arrêté des comptes.
Le Compte d'exploitation des projets de développement et entités assimilées récapitule en liste, les charges sans amortissement, ni dépréciation et une quotité des ressources équivalant aux charges pour que le solde des opérations de l'exercice soit nul.
Le Tableau des flux de trésorerie retrace les mouvements « entrées » et « sorties » de liquidités de l'exercice.
Le Tableau emplois-ressources récapitule tous les emplois, immobilisations et charges, sans amortissement ni dépréciation ainsi que les fonds reçus des bailleurs :
Le Tableau d'exécution budgétaire, fait apparaître le budget de l'exercice et les réalisations de l'exercice :
Le Tableau de réconciliation de trésorerie retrace les mouvements de trésorerie du début à la fin de l'exercice :
Les Notes annexes complètent et précisent l'information donnée par les autres états financiers annuels.

Article 8
Le Bilan de l'exercice fait apparaître de façon distincte :
1) à l’actif : l'actif immobilisé, l'actif circulant, la trésorerie-actif et l'écart de conversion-actif ;
2) au passif : les ressources stables, le passif circulant, la trésorerie-passif et l'écart de conversion-passif.

Article 9
Le Compte de résultat de l'exercice fait apparaître au crédit les produits et, au débit, les charges distinguées selon leur nature.
Pour les projets de développement, le Compte d'exploitation fait apparaître au débit les charges sans amortissement ni dépréciation et, au crédit, une quotité des ressources équivalant au total des charges pour obtenir un solde de l'exercice nul. 

Article 10
Le Tableau des flux de trésorerie de l'exercice fait apparaitre la trésorerie nette en début d'exercice, les flux de trésorerie provenant des activités opérationnelles, les flux de trésorerie provenant des opérations d'investissement, les flux de trésorerie provenant des fonds propres.
Les flux de trésorerie provenant des fonds extérieurs et la trésorerie nette en fin d'exercice.

Article 11
Le Tableau emplois-ressources, fait apparaitre les emplois (immobilisations et charges) sans amortissement ni dépréciation, les fonds reçus, l'excédent ou le déficit des fonds reçus sur les emplois, le montant de l'encaisse disponible.

Article 12
Le Tableau d'exécution budgétaire fait apparaître le budget de l'exercice, les décaissements. Les engagements non encore payés, les réalisations, le crédit disponible du budget et l'exécution du budget en valeur relative.
Article 13
Le Tableau de réconciliation de trésorerie retrace la trésorerie de début d'exercice, les transferts de fonds reçus des bailleurs, les emplois de l'exercice, la trésorerie de fin d'exercice et les paiements en instance.

Article 14
Le livre d'inventaire est un document obligatoire sur lequel sont transcrits :
1) pour les associations et les ordres professionnels, le Bilan, le Compte de résultat et le Tableau des flux de trésorerie de chaque exercice ainsi que le résumé de l'opération d'inventaire ;
2) pour les entités ayant pour objet la gestion ou l'administration de projets de développement, le Tableau emplois-ressources, le Tableau d'exécution budgétaire, le Tableau de réconciliation de trésorerie, le Bilan, le Compte d'exploitation de chaque exercice ainsi que le résumé de l'opération d'inventaire.

Article 15
Les états financiers annuels, décrits aux articles 7 à 13, sont accompagnés de Notes annexes, organisées par une référence croisée avec l'information liée. 
Les Notes annexes contiennent des informations complémentaires à celles qui sont présentées dans le Bilan, le Compte de résultat, le Compte d'exploitation, le Tableau des flux de trésorerie, le Tableau emplois-ressources, le Tableau d'exécution budgétaire, le Tableau de réconciliation de trésorerie. Les Notes annexes fournissent des descriptions narratives ou des décompositions d'éléments présentés dans les autres états financiers, ainsi que des informations relatives au% éléments qui ne répondent pas aux critères de comptabilisation dans les autres états financiers, Les Notes annexes comportent tous les éléments à caractère significatif qui ne sont pas mis en évidence dans les autres états financiers et sont susceptibles d'influencer le jugement que les utilisateurs des documents peuvent porter sur le patrimoine, la situation financière et la performance de l’entité, il en est ainsi notamment pour le montant des engagements donnés ct reçus dont le suivi doit être assuré par l'entité dans le cadre de son organisation comptable
Toute modification dans la présentation des états financiers annuels ou dans les méthodes d'évaluation doit être signalée dans les Notes annexes

Article 16
Les états financiers annuels de chaque entité respectent les dispositions ci-dessous :
1) le recours, pour la tenue de la comptabilité de l'entité, à un plan de comptes normalisé dont la liste figure dans le Système comptable des entités à but non lucratif ;
2) la tenue obligatoire de livres ou autres supports autorises ainsi que la mise en place de procédures nécessaires à organisation comptable permettant un contrôle interne une fiable et le contrôle externe par l'intermédiaire, Ie cas échéant, de l'auditeur, de la réalité des opérations ainsi que de la qualité des comptes, tout en favorisant la collecte des informations ;
3) A la clôture de chaque exercice, les organes d'administration ou de direction, selon le cas, dressent I ‘inventaire et les états financiers conformément aux dispositions de l`Acte uniforme et établissent un rapport d'activité ; le rapport d'activité expose la situation de l'entité durant l'exercice écoulé, ses perspectives de développement ou son évolution prévisible et l'évolution de la situation de trésorerie ; les événements importants. survenus entre la date de clôture de l'exercice et la date à laquelle il est établi, doivent également y être mentionnés ;
4) le Bilan d'ouverture d'un exercice doit correspondre au Bilan de clôture de l'exercice précédent ;
5) toute compensation, non juridiquement fondée, entre postes d'actif et postes de passif dans le Bilan et entre postes de charges et postes de produits dans le Compte de résultat est interdite :
6) la présentation des états financiers est identique d'un exercice à l'autre :
7) chacun des postes des états financiers comporte l'indication du chiffre relatif au poste correspondant de l'exercice précédent.
Lorsque l'un des postes chiffrés d'un état financier n'est pas comparable à celui de l'exercice précédent, c'est ce dernier qui doit être adapté. L'absence de comparabilité ou l'adaptation des chiffres est signalée dans les Notes annexes.
Pour la première année d'application, l'entité n'a l'obligation de renseigner que la colonne
N-l du Bilan.

Chapitre 3 - Moyens de contrôle
Article 17
Il est établi pour chaque entité à but non lucratif un registre des donateurs pour tous les dons, donations et legs reçus par l'entité.
Le registre des donateurs est coté, paraphé et numéroté de façon continue par la juridiction compétente de chaque État partie concerné.
Le registre des donateurs contient :
1) la date de l'opération :
2) les nom et prénoms, le domicile et l'adresse électronique des personnes physiques donatrices ;
3) la dénomination, le numéro d'immatriculation, le numéro d'identification fiscale, l'adresse du siège social et l'adresse électronique des personnes morales donatrices :
4) le montant et le mode de libération du don/legs mis à la disposition de l'entité à but non lucratif en espèces, par chèque, par virement ou en nature.
Toutes les écritures contenues dans ce registre doivent être signées par le représentant légal de l'entité à but non lucratif.
Ce registre peut être tenu en version physique reliée, brochée ou en version électronique.

Article 18
L'entité tient à jour le registre des donateurs.
S'il existe un auditeur, ce dernier soumet, à l'assemblée générale ou l'instance qui en tient lieu aux membres ou au bailleur de fonds du projet, un rapport qui constate l'existence du registre des donateurs et donne son avis sur sa tenue conforme.
S'il n'existe pas d'auditeur, une déclaration des dirigeants attestant de la tenue conforme du registre des donateurs est annexée audit rapport ou soumise à l'assemblée générale ou l'instance qui en tient lieu.

Article 19
Toute entité à but non lucratif est tenue de désigner au moins un auditeur lorsqu'elle remplit, à la clôture de l'exercice, l'un des trois critères suivants :
1) un total du bilan supérieur à cent millions (100.000.000) de francs CFA ou l'équivalent dans l'unité monétaire ayant cours légal dans l'Etat partie :
2) des ressources annuelles supérieures à deux cent millions (200.000.000) de francs CFA ou l'équivalent dans l'unité monétaire ayant cours légal dans l'État partie :
3) un effectif permanent supérieur à vingt (20) personnes ;
L'entité n'est plus tenue de désigner un auditeur dès lors qu'elle ne remplit plus aucun des trois
(3) critères fixés ci-dessus pendant les deux (2) exercices qui précédent l'expiration du mandat de l'auditeur.
Pour les autres entités à but non lucratif ne remplissant pas ces critères, la nomination de l'auditeur est facultative. Elle peut toutefois être demandée en justice par au moins dix-pour-cent (10%) des membres de l'entité.
Les états financiers et le rapport de gestion annuels sont transmis à l'auditeur s'il en a été désigné, quarante-cinq (45) jours au moins avant la date de l'assemblée générale ordinaire ou de l'instance qui en tient lieu de l'association et l'ordre professionnel, ou la date de transmission du rapport de l'auditeur aux bailleurs de fonds et/ou à l'Etat bénéficiaire du Projet de développement.
Dans les entités qui désignent, volontairement ou obligatoirement, un auditeur, ce dernier 1) soit, émet une opinion indiquant que les états financiers sont réguliers et sincères el donnent une image fidèle du résultat des opérations de l'exercice écoulé ainsi que de la situation financière et du patrimoine à la fin de cet exercice :
2) soit, exprime, en la motivant, une opinion avec réserve ou défavorable, ou indique qu'il est dans l'impossibilité d'exprimer une opinion.
L'auditeur se prononce sur la sincérité et la concordance des informations données dans le rapport d'activité avec les états financiers.

Article 20
L'auditeur est choisi par les membres de l'entité à but non lucratif parmi les experts-comptables inscrits au tableau de l'ordre des experts-comptables ou de l'organe qui en tient lieu dans chaque
Etat partie.

Article 21
L'auditeur est nommé pour trois (3) exercices renouvelables une fois. Toutefois, si l'entité a une existence inférieure à trois exercices, son mandat est ramené à cette durée.
L'auditeur est nommé par l'assemblée générale de l'entité ou l'instance qui en tient lieu à la majorité de ses membres représentant au moins plus de la moitié des membres présents ou représentés, ou par le bailleur de fonds ct/ou l'État partie bénéficiaire du Projet de développement.
Si la majorité ci-dessus n'est pas obtenu et sauf clause contraire des statuts, l'assemblée générale de l'entité ou l'instance qui en tient lieu réunie sur deuxième convocation peut valablement désigner l'auditeur lorsque le quorum d'un dixième (1/10) des membres présents ou représentés est atteint.
Si le quorum ci-dessus n'est pas atteint sur deuxième convocation ou lorsque l'assemblée générale de l'entité ou l'instance qui en tient lieu ne procède pas à la nomination d'un auditeur, tout membre peut demander à la juridiction compétente la désignation d'un auditeur.

Article 22
Si l'assemblée ou l'instance qui en tient lieu ne procède pas au renouvellement du mandat de l'auditeur ou à son remplacement à l'expiration de son mandat, la mission de l'auditeur est prorogée, sauf refus exprès de sa part. 
La prorogation de la mission de l'auditeur prévue à l'alinéa 1° du présent article s'opère jusqu'à la plus prochaine assemblée générale ou à la prochaine réunion de l'instance qui en tient lieu statuant sur les comptes de l'entité, ou à la prochaine approbation des comptes du projet par le bailleur de fonds ou l'État partie.

Article 23
Il est procédé régulièrement, par voie de Décision du Conseil des ministres de l'OHADA, à la mise à jour du Système comptable des entités à but non lucratif, sur recommandation de la Commission de normalisation pour la profession comptable conformément au Règlement portant création, organisation et fonctionnement de ladite Commission.

Chapitre 4 - Dispositions pénales
Article 24
Encourent une sanction pénale les dirigeants des entités à but non lucratif qui :
• n'ont pas, pour un exercice, dressé l'inventaire et établi les états financiers annuels, ainsi que le rapport d'activité :
• ont sciemment établi et communiqué des états financiers qui ne donnent pas une image fidèle du patrimoine, de la situation financière et du résultat de l'exercice ;
• n'ont pas tenu et mis à jour le registre des donateurs.

Article 25
Encourent une sanction pénale, les dirigea ts d'entités à but non lucratif qui n'ont pas provoqué la désignation de l'auditeur de l'entité ou ne l'ont pas convoqué à l'assemblée générale ou à la réunion de l'instance qui en tient lieu statuant sur les comptes de l'entité.

Article 26
Encourent une sanction pénale, les dirigeants d'entités à but non lucratif ou toute personne au service de l'entité qui, sciemment, ont fait obstacle aux vérifications ou au contrôle de l'auditeur, ou qui ont refusé la communication sur place de toutes les pièces utiles à l'exercice de sa mission, notamment les contrats, livres, documents comptables et registres.

Article 27
Les infractions prévues par le présent Acte uniforme sont punies conformément aux dispositions du droit pénal en vigueur dans chaque État partie. 

Article 28
Le présent Acte uniforme, auquel est annexé le système comptable des entités à but non lucratif, sera publié au Journal Officiel de l'OHADA dans un délai de soixante (60) jours à compter de la date de son adoption. Il sera également publié dans les États parties au Journal Officiel ou par tout autre moyen approprié.
Le présent Acte uniforme est applicable à compter du 1er janvier 2024.



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
