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


LOI_ASSOCIATIONS_BENIN = """
====================================================================
LOI N° 2025-19 DU 22 JUILLET 2025
relative aux associations et aux fondations en République du Bénin
L'Assemblée nationale a délibéré et adopté en sa séance du 09 juillet 2025.
Le Président de la République promulgue la loi dont la teneur suit :
====================================================================

========================================
TITRE PREMIER
DISPOSITIONS GÉNÉRALES COMMUNES AUX ASSOCIATIONS ET AUX FONDATIONS
========================================

--- CHAPITRE I : DÉFINITIONS ---

Article 1er : Au sens de la présente loi, les termes ci-après sont définis comme suit :
- association : convention par laquelle deux ou plusieurs personnes physiques ou morales mettent en commun, d'une façon autonome et permanente, leurs connaissances, leurs moyens ou leurs activités, dans un but autre que de partager des bénéfices.
- association artistique et culturelle : association poursuivant des objectifs se rattachant aux arts et à la culture.
- association de base : association constituée, non membre d'une autre.
- association étrangère : association constituée soit suivant des lois autres que celles de la République du Bénin, soit selon le droit béninois mais dont le siège est à l'étranger ou les fondateurs sont en majorité de nationalité étrangère.
- association professionnelle : association constituée sur la base de l'appartenance à une même profession ou à une profession similaire, chargée d'assurer la promotion et la défense des intérêts professionnels collectifs des membres.
- association reconnue d'utilité publique : association ayant un but d'intérêt général et reconnue comme telle par l'État.
- confédération d'associations : regroupement d'organisations faîtières d'associations pour la poursuite d'objectifs communs.
- consortium d'associations : regroupement d'associations ou d'organisations faîtières d'associations, qui décident de partager des ressources matérielles, immatérielles, humaines et financières pour atteindre un objectif précis.
- fédération d'associations : regroupement d'associations de base pour la poursuite d'objectifs communs, doté de la personnalité juridique.
- fondation : personne morale de droit privé ou de droit public à but non lucratif créée par un ou plusieurs donateurs, eux-mêmes pouvant être des personnes physiques ou morales pour accomplir une œuvre d'intérêt général.
- fondation reconnue d'utilité publique : fondation à laquelle la reconnaissance d'utilité publique est accordée par l'État.
- fondation d'entreprise : fondation créée en vue de la réalisation d'une œuvre d'intérêt général par une ou plusieurs personnes morales, à savoir sociétés civiles ou commerciales, établissements publics, coopératives, mutuelles.
- organisation non gouvernementale (ONG) : regroupement de personnes physiques ou morales, de nationalité béninoise ou étrangère, constituant une organisation indépendante des États et des institutions internationales, en vue d'exercer une activité d'intérêt général, de solidarité ou de coopération volontaire pour le développement.
- réseau d'associations : regroupement formel d'organisations faîtières d'associations qui ont toutes un objet proche ou qui œuvrent ensemble dans un même but.

--- CHAPITRE II : OBJET - CHAMP D'APPLICATION ---

Article 2 : La présente loi régit l'exercice de la liberté d'association et les conditions d'exercice des activités de certains organismes de générosité concourant aux œuvres d'intérêt général. Elle détermine notamment les conditions et modalités de création, d'organisation, de fonctionnement et de contrôle des associations et des fondations. Elle régit également les organisations non gouvernementales créées sous forme d'association ou de fondation.

Article 3 : La présente loi ne s'applique pas aux associations dont la création est régie par des dispositions législatives particulières, notamment les partis politiques, les syndicats, les organisations de la chefferie traditionnelle.

Article 4 : En dehors des dispositions visant expressément les associations non déclarées, les termes « association » ou « associations » visent dans la présente loi les associations ou organisations non gouvernementales légalement constituées et déclarées.

--- CHAPITRE III : PRINCIPES GÉNÉRAUX ---

Article 5 : L'État prend les mesures nécessaires en vue :
- de garantir l'exercice et la jouissance de la liberté d'association ;
- d'assurer la libre adhésion de toute personne physique ou morale de nationalité béninoise ou étrangère à l'association de son choix, dans les conditions fixées par les lois et les règlements ;
- d'encourager et de valoriser la contribution des associations au développement de la nation.

Article 6 : L'État et les collectivités territoriales favorisent la promotion des associations, des fondations et des organisations non gouvernementales, dans les conditions fixées par les lois et les règlements.

Article 7 : Les actions d'appui et de contrôle de l'État et des collectivités territoriales respectent les principes d'autonomie de gestion des associations et des fondations.

Article 8 : L'adhésion à une association est libre. Nul ne peut être contraint à adhérer à une association.

Article 9 : Les personnes physiques non majeures ne peuvent créer une fondation, créer ou être membres d'une association que par représentation dans les conditions déterminées par la loi.

Article 10 : Des étrangers régulièrement installés sur le territoire de la République du Bénin peuvent librement créer entre eux, ou ensemble avec des Béninois, une association. Ils peuvent créer une fondation dans les mêmes conditions. Ils peuvent adhérer à une association ou affecter certains de leurs biens à une fondation.

Article 11 : Nul ne peut faire l'objet de discrimination ou de mesures portant atteinte à ses droits reconnus par la Constitution ayant pour motif son appartenance à une association ou sa contribution à une fondation légalement constituée et enregistrée.

Article 12 : Tout membre d'une association peut s'en retirer à tout moment, après s'être acquitté de toutes les obligations qui lui incombent.

Article 13 : Les membres de toute association pris en cette qualité, jouissent de la liberté d'expression et d'opinion, de réunion, conformément aux lois, règlements et statuts de ladite association.

Article 14 : Toute association ou regroupement d'associations a un objet licite. L'objet de toute association, sa dénomination, le lieu de son siège, les droits et obligations des membres sont fixés par ses statuts et ne peuvent être modifiés qu'en assemblée générale des membres, dans les conditions prévues par les statuts, sous réserve des dispositions particulières de la présente loi.

Article 15 : Toute association ou fondation, ayant un objet ou une cause contraire aux lois et aux bonnes mœurs, est nulle et de nul effet. La nullité d'une association ou d'une fondation ne peut être constatée que par une juridiction compétente saisie par toute personne ayant intérêt.

Article 16 : Sont illicites les associations ou fondations prônant ou ayant des pratiques contraires à la dignité de la personne humaine, notamment la haine, l'intolérance, la xénophobie, le racisme, la torture ou le terrorisme. Le caractère illicite d'une association ou d'une fondation ne peut être constaté que par une juridiction compétente saisie par toute personne ayant intérêt.

Article 17 : Toute association ou toute fondation a un siège. Ce siège est fixé par ses statuts. Il ne peut être fictif. Il ne peut être constitué uniquement par une domiciliation à une boîte postale ou une boîte électronique. Il est établi à une adresse géographique précise.

Article 18 : Toute association peut s'affilier librement à un cadre de concertation ou à une organisation faîtière et s'en retirer librement.

Article 19 : Les associations fonctionnent sur le principe démocratique qui se concrétise à travers la garantie des droits des membres, les pouvoirs et le fonctionnement de l'assemblée générale.

Article 20 : Les règles établies par la présente loi pour les associations s'appliquent à leurs organisations faîtières.

Article 21 : Lorsqu'une association ou une fondation bénéficie d'un financement public ou d'une aide comportant une part provenant de l'État, de ses démembrements ou d'une collectivité territoriale, elle établit et soumet à ceux-ci, dans un délai raisonnable, un rapport sur les activités concernées par ce financement et l'utilisation des fonds alloués, sans préjudice des dispositions légales en matière de vérification des comptes et de contrôle de la gestion par les juridictions financières.

Article 22 : Toute association ou fondation déclare à l'autorité chargée de la tenue du registre des associations et fondations, toute ressource financière reçue de toute institution privée nationale ou internationale. Elle lui produit également, pour ces ressources, le rapport visé à l'article précédent de la présente loi.

Article 23 : Toute association ou fondation tient une comptabilité conformément aux textes en vigueur.

Article 24 : Il est institué un registre des associations et fondations destiné à recevoir les déclarations d'existence, les inscriptions modificatives les concernant ainsi que toutes autres déclarations prescrites par les lois et règlements pour y être mentionnées. Le registre peut être tenu en version électronique. Les mentions du registre sont fixées par décret pris en Conseil des ministres.

========================================
TITRE II
DISPOSITIONS PARTICULIÈRES APPLICABLES AUX ASSOCIATIONS
========================================

--- CHAPITRE I : CONSTITUTION - EXISTENCE JURIDIQUE DES ASSOCIATIONS ---

SECTION 1 — CONSTITUTION DES ASSOCIATIONS

Article 25 : Les associations se constituent librement sans autorisation administrative préalable. Elles sont régies quant à leur formation et à leur validité par les principes généraux du droit des contrats.

Article 26 : Lorsque l'objet d'une association ou les activités qui en découlent donne lieu à une règlementation particulière, leur constitution respecte ladite règlementation dans la mesure prescrite.

Article 27 : Seules peuvent avoir la qualité de membres fondateurs ou d'adhérents d'une association :
- les personnes physiques jouissant de leurs droits civiques ;
- les personnes morales légalement constituées.

Article 28 : Les membres fondateurs ou adhérents d'une association condamnés à une peine avec perte de leurs droits civiques perdent de plein droit leur qualité de membre. Les membres d'une association condamnés à une peine correctionnelle devenue définitive ne peuvent être désignés dans les organes dirigeants qu'après avoir purgé leur peine ou bénéficié d'une grâce ou d'une remise. Les membres d'une association condamnés à une peine criminelle devenue définitive ne peuvent être désignés dans les organes dirigeants qu'après avoir été amnistiés ou réhabilités dans les conditions prescrites par les lois et règlements.

Article 29 : Sous réserve des dispositions de la présente loi, les personnes désirant constituer une association ou une organisation faîtière d'associations sont tenues :
- d'organiser une assemblée constitutive qui en adopte les statuts et règlement intérieur ;
- d'établir un procès-verbal de l'assemblée constitutive.
Les mentions obligatoires des statuts, du règlement intérieur et du procès-verbal sont précisées par décret pris en Conseil des ministres.

Article 30 : Les statuts d'une association peuvent être rédigés sous seing-privé ou sous forme notariée. Lorsque les statuts d'une association sont dressés sous la forme d'un acte sous seing-privé, il est établi autant d'originaux qu'il est nécessaire pour le dépôt d'un exemplaire au siège de l'association et l'exécution des formalités de constitution prescrites par la loi.

Article 31 : Toute association définit librement dans ses statuts, son objet, ses objectifs, son siège, la sphère géographique de ses activités, les modalités de son fonctionnement, et notamment les organes de gouvernance, les conditions d'adhésion et de retrait des membres, leurs obligations et les règles de discipline collective, sans préjudice aux dispositions des lois et règlements.

Article 32 : Les associations peuvent agir ou interagir dans tout domaine d'activités visant le développement de la personne humaine, dans toutes ses dimensions, notamment culturelle, sociale, religieuse, économique, scientifique et environnementale.

SECTION 2 — EXISTENCE - STATUT JURIDIQUES DES ASSOCIATIONS

Article 33 : Les associations légalement constituées et déclarées acquièrent la personnalité juridique. Les associations qui ne sont pas légalement constituées et qui n'ont pas accompli la formalité de déclaration d'existence ne peuvent prétendre au statut et aux droits d'une association déclarée au sens de la présente loi.

Article 34 : Les associations soumises à la présente loi sont des organisations apolitiques.

Article 35 : Toute association accomplit les formalités de déclaration d'existence au registre des associations et fondations. La liste des pièces constitutives du dossier de déclaration d'existence au registre des associations et fondations est fixée par décret pris en Conseil des ministres.

Article 36 : La déclaration d'existence au registre des associations et fondations est constatée par un récépissé délivré par l'autorité administrative compétente désignée par décret pris en Conseil des ministres.

Article 37 : Toute association jouit de la personnalité juridique à compter de la date de délivrance de son récépissé de déclaration d'existence au registre des associations et fondations.

Article 38 : Il est institué un Journal du registre des associations et fondations destiné à la publication :
- des récépissés de déclaration d'existence au registre des associations et fondations ;
- des inscriptions modificatives et autres publications relatives aux associations et aux fondations prescrites par les lois et règlements.
Les publications au Journal du registre des associations et fondations sont opposables aux tiers à compter de leurs dates d'inscription sur le registre. L'autorité chargée de la tenue du registre des associations et fondations tient le Journal du registre des associations et fondations. Les frais des publications au Journal du registre des associations et fondations sont fixés par décision de l'autorité chargée de la tenue du registre. Le Journal du registre des associations et fondations est cessible par exemplaire à toute personne intéressée.

Article 39 : Le récépissé de déclaration d'existence au registre des associations et fondations est délivré après vérification de la conformité du dossier de déclaration aux lois et règlements. A la réception d'une déclaration d'existence au registre des associations et fondations, il est délivré séance tenante au déposant, une attestation de dépôt comportant la dénomination de l'association, son objet, son adresse et le numéro d'enregistrement.

Article 40 : L'autorité compétente dispose d'un délai maximum de soixante jours pour procéder à la vérification de conformité de toute déclaration d'existence au registre des associations et fondations et pour délivrer le récépissé de déclaration d'existence. En cas de silence de l'autorité administrative compétente à l'expiration du délai de soixante jours prévu au premier alinéa du présent article, le dossier est réputé conforme et le récépissé doit être délivré. Toutes observations ou compléments de pièces nécessaires à la régularité du dossier sont notifiés dans le délai prévu à l'alinéa 1er du présent article au déposant. Le délai pour procéder à la vérification de conformité et délivrer le récépissé de déclaration d'existence court à nouveau à compter de la production des pièces requises ou de la preuve de la satisfaction aux observations de l'autorité compétente.

Article 41 : L'autorité administrative compétente ne peut refuser la délivrance du récépissé de déclaration d'existence au registre des associations et fondations que pour des motifs de légalité dont elle précise expressément les fondements.

Article 42 : La décision de refus de délivrance du récépissé de déclaration d'existence au registre des associations et fondations est susceptible de recours pour excès de pouvoir dans les conditions prévues par les règles de procédures applicables devant la juridiction compétente.

Article 43 : Le premier responsable de l'organe ayant pour fonction d'assurer la représentation d'une association est tenu de faire une déclaration modificative au registre des associations et fondations, dans un délai de trente jours, de tous changements survenus dans la composition des organes dirigeants et de toutes modifications apportées aux statuts. En cas de défaut de déclaration modificative, les changements ou modifications sont inopposables aux tiers. En cas de défaut d'inscription, les changements ou modifications sont nuls et de nul effet.

Article 44 : Toute personne a le droit de prendre copie, à ses frais, auprès de l'autorité administrative compétente, des statuts de toute association déposés au registre des associations et fondations.

--- CHAPITRE II : ORGANISATION - FONCTIONNEMENT - DROITS - OBLIGATIONS DES ASSOCIATIONS ---

SECTION 1 — RÈGLES GÉNÉRALES D'ORGANISATION ET DE FONCTIONNEMENT

Article 45 : Toute association dispose, quelle que soit sa dénomination, au moins d'un organe délibérant, cadre d'expression de tous ses membres et d'un organe exécutif.

Article 46 : Toute association fonctionne suivant des principes démocratiques établis par ses statuts. Les organes qui assurent l'administration, la direction et la gestion, à l'exception du personnel salarié, font l'objet d'un renouvellement périodique conformément aux statuts.

Article 47 : Les membres des organes dirigeants exercent leurs mandats avec intégrité et transparence. Ils soumettent, chaque année à l'approbation de leurs membres, dans les conditions prévues par les statuts, un rapport annuel sur les comptes et les activités.

Article 48 : Les membres d'une association ont droit à l'information sur toutes les activités programmées et mises en œuvre au nom de celle-ci, notamment les projets et programmes qu'elle exécute, les fonds qu'elle collecte ou qu'elle reçoit ainsi que sur la gestion administrative et financière en général.

SECTION 2 — OBLIGATIONS DES ASSOCIATIONS

Article 49 : Toute association respecte les lois, les règlements, les conventions, accords et traités dûment ratifiés par la République du Bénin.

Article 50 : Toute association a le devoir de contribuer à la préservation, à la restauration, et au maintien de la paix ainsi qu'à la promotion du vivre ensemble entre les citoyens. Elle contribue à la culture de la bonne gouvernance et du respect de la chose publique. En conséquence, il est interdit à toute association, dans ses activités, de prendre des positions politiques de susciter ou d'encourager tout acte contraire aux lois et règlements, notamment toute forme de violence, de discrimination, d'injure et de sédition. Tout manquement par l'un quelconque des membres d'une association à l'obligation prévue à l'alinéa précédent est puni d'un emprisonnement de six mois à un an et d'une amende d'un million (1 000 000) de francs CFA ou de l'une de ces deux peines seulement.

Article 51 : Les organes dirigeants de toute association s'assurent que ses programmes et activités sont conformes à son objet et ses objectifs mentionnés dans ses statuts. L'autorité compétente peut, en cas de nécessité, procéder à la vérification de conformité des activités à l'objet de l'association et, en cas de non-conformité, ordonner les mesures de remédiation nécessaires.

Article 52 : Les organes dirigeants de toute association publient au Journal du registre des associations et fondations, au plus tard le 30 avril de chaque année, un rapport général sur l'année écoulée indiquant notamment ses programmes, ses ressources, l'état d'exécution de ses activités et programmes et ses perspectives.

SECTION 3 — DROITS DES ASSOCIATIONS

Article 53 : Toute association jouit de la liberté d'expression, de la liberté de réunion, de manifestation pacifique et du droit d'accès à l'information sur les affaires publiques, dans le respect des textes en vigueur. Elles peuvent se prononcer sur tout sujet d'intérêt général.

Article 54 : Toute association peut ester en justice pour défendre ses intérêts et ceux de ses membres.

Article 55 : Aux fins du financement de ses activités, toute association peut rechercher et accepter, sous réserve des interdictions ou restrictions déterminées par décret pris en Conseil des ministres, des dons et legs, de personnes physiques ou morales privées, de nationalité béninoise ou étrangère, membres ou non de l'association. Toute association peut en outre bénéficier :
- des appuis de l'État, des collectivités territoriales et de toute personne morale de droit public ;
- de régimes particuliers d'exonérations ou de réductions fiscales.
Au début et à la fin de chacune des activités à caractère public des associations bénéficiant de financements d'États étrangers, de personnes morales étrangères ou de personnes physiques non-résident sur le territoire de la République du Bénin, leurs organes dirigeants ou représentants mentionnent expressément à l'attention du public, l'origine de leurs financements étrangers inclus dans le budget de l'année de l'activité.

Article 56 : Dans le cadre de la poursuite de ses objectifs, toute association peut exercer, à titre accessoire, une activité économique génératrice de profits à condition que ces profits ne soient pas distribués directement ou indirectement, même partiellement, entre ses membres.

Article 57 : Les dirigeants de toute association ainsi que ses membres participent aux activités de l'association à titre essentiellement bénévole. Toutefois, le caractère bénévole ne fait pas obstacle à ce que, en raison de sujétion particulière, l'association leur accorde des avantages soit en nature, soit en numéraires qui ne représentent pas la contrepartie exacte de leurs prestations.

Article 58 : Toute association peut employer un personnel salarié, conformément aux textes en vigueur.

--- CHAPITRE III : DISSOLUTION - DÉVOLUTION DES BIENS DES ASSOCIATIONS ---

Article 59 : Toute association peut être dissoute conformément à ses statuts ou par décision de justice.

Article 60 : L'autorité administrative compétente peut suspendre les activités de l'association, s'il est établi que cette association exerce une activité ayant une cause ou un objet illicite, se livre à des activités contraires à ses statuts ou aux lois et règlements. Lorsque l'autorité administrative prononce une décision de suspension dans les conditions prévues à l'alinéa 1 du présent article, elle saisit le tribunal compétent pour statuer sur la dissolution de l'association. La décision de suspension est susceptible de recours pour excès de pouvoir dans les conditions prévues par les règles de procédures applicables devant la juridiction compétente.

Article 61 : La dissolution de toute association peut être demandée, par toute personne qui y a intérêt, devant le tribunal de première instance du lieu du siège de l'association statuant en matière civile. Le tribunal saisi statue en procédure d'urgence dans un délai de trente jours à compter de sa saisine. Il prescrit aux parties dans ce délai les diligences qui leur incombent et fixe les délais qu'elle juge convenable pour les accomplir. Le tribunal peut aussi soit à la requête du demandeur, soit du ministère public, ordonner la suspension des activités de l'association. La décision est exécutoire par provision.

Article 62 : La décision de la juridiction compétente relative à la dissolution ou à la suspension des activités d'une association est susceptible de recours dans les conditions de droit commun.

Article 63 : Lorsque la dissolution d'une association est prononcée conformément à ses statuts, il est nommé, conformément auxdits statuts, un ou plusieurs liquidateurs chargés du recouvrement des créances et de l'apurement des dettes de l'association au besoin par cession de biens. Ces opérations ne peuvent excéder six mois à compter de la date de nomination du ou des liquidateurs. A défaut de clôture des opérations dans ce délai, il est procédé à la poursuite et à la clôture de ces opérations par un liquidateur désigné par voie de justice.

Article 64 : Lorsqu'une association est dissoute conformément à ses statuts et quelle que soit la manière dont il est procédé aux opérations visées à l'article 63 de la présente loi, l'organe dirigeant compétent de l'association demeure compétent pour décider de la dévolution des biens restants après lesdites opérations. A défaut pour cet organe d'y avoir procédé, dans un délai de trente jours à compter de la clôture des opérations visées à l'article 63 de la présente loi, il y est procédé par voie de justice à la diligence de toute personne qui y a intérêt.

Article 65 : Lorsque la dissolution d'une association est prononcée par décision de justice, les biens de l'association sont dévolus à toute entité publique susceptible de les recevoir, après apurement des dettes au besoin par cession de biens.

Article 66 : Sont punis d'un emprisonnement de six mois à un an et d'une amende d'un million (1 000 000) de francs CFA ou de l'une de ces deux peines seulement, les membres fondateurs, directeurs ou administrateurs d'une association dissoute par décision de justice qui poursuivraient leurs activités malgré la dissolution de l'association ou qui aurait reconstitué illégalement l'association après une décision judiciaire de dissolution. Sont punis des mêmes peines, tous ceux qui auront favorisé la poursuite des activités ou la reconstitution illégale prévues visée au premier alinéa du présent article.

--- CHAPITRE IV : RÉGIMES SPÉCIAUX ---

SECTION 1 — RÈGLES PARTICULIÈRES AUX ASSOCIATIONS ÉTRANGÈRES

Article 67 : Toute association étrangère dotée de la personnalité juridique qui désire exercer des activités en République du Bénin sollicite une autorisation préalable. L'autorisation est délivrée par l'autorité chargée de la tenue du registre des associations et fondations, dans les conditions fixées par décret pris en Conseil des ministres. Elle est enregistrée au registre des associations et fondations. L'autorisation est publiée au Journal du registre des associations et fondations, aux frais de l'association. La décision de refus d'autorisation est susceptible de recours pour excès de pouvoir dans les conditions prévues par les règles de procédures applicables devant la juridiction compétente.

Article 68 : Toute association étrangère autorisée jouit de la personnalité juridique en République du Bénin. L'association étrangère autorisée jouit des droits et est tenue par les obligations prévues par la présente loi pour toute association non étrangère, sans qu'il n'y ait lieu à la modification de ses statuts en ce qui concerne ses organes de décision et ses règles de fonctionnement.

Article 69 : Toute association étrangère autorisée peut signer avec la République du Bénin un accord de siège qui lui confère privilèges consulaires ou diplomatiques pendant la durée de validité de l'accord. Les conditions, la procédure de conclusion ainsi que les avantages conférés dans le cadre des accords de siège sont précisés par décret pris en Conseil des ministres.

Article 70 : Les activités de toute association étrangère peuvent être suspendues par décision de l'autorité administrative compétente en cas de manquement aux lois et règlements et à ses statuts.

Article 71 : L'autorisation de toute association étrangère peut être retirée par décision de l'autorité administrative compétente en cas de manquement grave aux lois et règlements.

Article 72 : Les décisions visées aux articles 67, 70 et 71 de la présente loi sont susceptibles de recours pour excès de pouvoir dans les conditions prévues par les règles de procédures applicables devant la juridiction compétente.

SECTION 2 — RÈGLES PARTICULIÈRES AUX ASSOCIATIONS RECONNUES D'UTILITÉ PUBLIQUE

Article 73 : Toute association exerçant ses activités en République du Bénin qui poursuit un but reconnu d'intérêt général peut être reconnue d'utilité publique.

Article 74 : La reconnaissance d'utilité publique est une décision par laquelle l'État reconnaît les objectifs poursuivis par une association comme concourant efficacement à la réalisation des politiques de développement de l'État, au regard des activités de ladite association sur une période déterminée.

Article 75 : Une association ne peut être reconnue d'utilité publique qu'après une période probatoire de cinq années d'activités consécutives constatées par des rapports d'activités établis dans le respect des dispositions des articles 21, 22 et 52 de la présente loi. Toutefois, lorsqu'il en est justifié, le délai visé au présent article peut être spécialement réduit par décision du Conseil des ministres. La reconnaissance d'utilité publique ne peut être accordée que si l'évaluation de l'impact des activités de l'association sur l'amélioration des résultats des politiques de développement de l'État dans un secteur déterminé est jugée satisfaisante ou si leur impact potentiel est considéré comme de nature à contribuer à une telle amélioration.

Article 76 : La demande de la reconnaissance d'utilité publique est adressée à l'autorité chargée de la tenue du registre des associations et fondations.

Article 77 : La reconnaissance d'utilité publique est décidée par décret pris en Conseil des ministres.

Article 78 : La procédure et les modalités de la reconnaissance d'utilité publique sont précisées par décret pris en Conseil des ministres.

Article 79 : Toute association reconnue d'utilité publique peut bénéficier d'une subvention annuelle de l'État. Elle peut bénéficier de la garantie de l'État ou d'une collectivité publique pour l'accès au financement de ses activités. Elle peut également bénéficier d'avantages fiscaux déterminés conformément aux dispositions des lois de finances ou aux stipulations des accords-cadres conclus avec l'État. Nonobstant les dispositions du premier alinéa du présent article, les associations reconnues d'utilité publique bénéficient au moins des avantages douaniers et fiscaux ci-après :
- exonération de l'impôt sur les sociétés ;
- exonération des droits et taxes à l'entrée y compris la taxe sur la valeur ajoutée sur :
  * les matériels et équipements ainsi que les véhicules importés en République du Bénin ou acquis sur place nécessaires à la mission de l'association ;
  * le matériel technique didactique ainsi que les ouvrages importés par l'association ;
  * les effets personnels importés par le personnel expatrié de l'association dans les six premiers mois de son installation ;
  * les dons et legs.

Article 80 : L'État conclut de plein droit un accord de siège avec toute association étrangère reconnue d'utilité publique. L'accord de siège stipule les avantages notamment douaniers et fiscaux accordés à l'association.

Article 81 : L'État ou une collectivité territoriale peut déléguer, par entente directe et sans appliquer les procédures de passation des marchés publics, la gestion d'un service public à une association reconnue d'utilité publique contre rémunération sur la base d'une convention de délégation de service public.

Article 82 : Toute association reconnue d'utilité publique peut ester en justice pour la défense de cause relevant de l'intérêt général. Les sommes reçues à titre de réparation dans ce cas sont versées au Trésor public.

Article 83 : Toute association reconnue d'utilité publique peut solliciter et bénéficier, pour une durée déterminée, de la part de l'État ou des collectivités territoriales, la mise à disposition de personnes possédant des compétences spécifiques pour les besoins de ses activités.

Article 84 : Toute association reconnue d'utilité publique peut faire appel public à la générosité des donateurs nationaux ou internationaux dans les conditions déterminées par décret pris en Conseil des ministres. Toute association reconnue d'utilité publique dispose d'un commissaire aux comptes qui exécute sa mission telle que prescrite par les textes en vigueur.

Article 85 : La reconnaissance d'utilité publique peut être retirée en cas de manquement grave par l'association à ses obligations, aux lois et règlements. La mauvaise gestion de ressources publiques, des fonds du public ou de financements garantis par l'État ou une collectivité territoriale est considérée comme un manquement grave aux obligations d'une association reconnue d'utilité publique. Le manquement grave prévu à l'alinéa 2 du présent article est puni d'un emprisonnement de six mois à un an et d'une amende d'un million (1 000 000) de francs CFA ou de l'une de ces deux peines seulement, sans préjudice de qualifications pénales plus graves.

Article 86 : La décision de retrait de la reconnaissance d'utilité publique est susceptible de recours pour excès de pouvoir dans les conditions prévues par les règles de procédures applicables devant la juridiction compétente.

SECTION 3 — RÈGLES PARTICULIÈRES AUX ASSOCIATIONS SIGNATAIRES D'UN ACCORD-CADRE AVEC L'ÉTAT

Article 87 : Au sens de la présente loi, un accord-cadre est un accord conclu entre l'État et une ou plusieurs associations pour définir les modalités générales de leur coopération.

Article 88 : Toute association reconnue au Bénin peut signer un accord-cadre avec le gouvernement de la République du Bénin.

Article 89 : Tout accord-cadre entre l'État et une association précise, notamment :
- les objectifs de l'accord-cadre ;
- les secteurs ou activités ciblés ;
- les modalités et sources de financement ;
- l'engagement de concourir à la réalisation de la politique de développement économique et social du gouvernement ;
- l'engagement de réaliser les activités ciblées dans des zones géographiques et domaines d'intervention ;
- l'engagement de recruter prioritairement du personnel de nationalité béninoise dans le cadre de la mise en œuvre des activités ;
- les avantages fiscaux et douaniers à l'association ayant signé l'accord cadre ;
- la durée de l'accord-cadre.

Article 90 : Une association ne peut signer un accord-cadre avec l'État qu'après une période probatoire de trois années d'activités consécutives constatées par des rapports d'activités établis dans le respect des dispositions des articles 21 et 52 de la présente loi. Toutefois, lorsqu'il en est justifié, le délai visé au présent article peut être spécialement réduit par décision prise en Conseil des ministres.

Article 91 : L'accord-cadre peut être signé à l'initiative soit de l'autorité publique sectorielle compétente, soit de l'association ou des associations concernées.

Article 92 : Les conditions, la procédure, les modalités de conclusion ainsi que les avantages conférés dans le cadre des accords-cadres sont précisés par décret pris en Conseil des ministres.

========================================
TITRE III
DISPOSITIONS PARTICULIÈRES APPLICABLES AUX FONDATIONS
========================================

--- CHAPITRE I : DISPOSITIONS COMMUNES À TOUTES LES FONDATIONS ---

Article 93 : Les publications des déclarations modificatives, des statuts des fondations, de tous changements survenus dans la composition des organes dirigeants et de toutes modifications, de toute décision de dissolution ainsi que toutes autres publications prescrites par la loi sont faites au Journal du registre des associations et fondations prévu par la présente loi, dans un délai de trente jours à compter de la date de l'acte ou de l'événement concerné, à défaut d'un délai spécifique autrement fixé par des dispositions particulières. A défaut de publication dans le délai prévu au premier alinéa du présent article, les actes concernés sont inopposables aux tiers.

SECTION 1 — CRÉATION - PERSONNALITÉ JURIDIQUE DES FONDATIONS

SOUS-SECTION 1 — CRÉATION

Article 94 : Une fondation est créée par une affectation irrévocable de biens, droits ou ressources, par une ou plusieurs personnes physiques ou morales de droit public ou privé, en vue de la poursuite d'un objectif qui peut être d'intérêt général. La fondation ne comprend pas de membres.

Article 95 : La création de toute fondation est constatée par l'adoption de ses statuts par la ou les personnes ayant consenti à la création et à lui affecter des biens, droits ou ressources. L'adoption des statuts de la fondation est constatée par un procès-verbal signé par le ou les fondateurs. Les statuts sont soumis aux exigences de contenu prescrites pour les statuts des associations.

Article 96 : La création d'une fondation par l'État, comme unique fondateur, est constatée par décret pris en Conseil des ministres.

Article 97 : Les causes d'intérêt général pour lesquelles une fondation peut être créée concernent tout domaine de la vie nationale.

Article 98 : L'appellation « fondation » ne peut être utilisée dans la dénomination d'une entité régie par la présente loi que par celle qui a été créée et déclarée comme fondation au registre des associations et fondations.

SOUS-SECTION 2 — EXISTENCE JURIDIQUE

Article 99 : Les dirigeants de toute fondation créée accomplissent les formalités de déclaration d'existence de la fondation.

Article 100 : L'accomplissement des formalités prévues pour les associations aux articles 35 à 42 de la présente loi confère à la fondation la personnalité juridique.

SOUS-SECTION 3 — QUALITÉ DE FONDATEUR D'UNE FONDATION

Article 101 : La personne ou les personnes ayant pris l'initiative de la création d'une fondation et ayant signé le procès-verbal constatant l'approbation de ses statuts ont la qualité de fondateurs historiques.

Article 102 : Les personnes qui, postérieurement à la création d'une fondation, lui ont apporté une contribution significative pour le développement de ses activités et l'atteinte de ses objectifs, peuvent se voir reconnaître, à leur demande, la qualité de fondateurs, par décision unanime des fondateurs historiques. Ils sont dénommés « fondateurs agréés ». Le caractère significatif de la contribution de la personne qui demande à acquérir la qualité de fondateur agréé est souverainement appréciée par les fondateurs historiques. Les statuts de la fondation peuvent préciser et compléter les critères de reconnaissance de la qualité de fondateur agréé.

Article 103 : Lorsqu'une fondation a été créée par l'État, comme unique fondateur, la décision d'accepter d'autres fondateurs postérieurement, est prise en Conseil des ministres. Dans ce cas, les statuts de la fondation sont modifiés au cours d'une réunion des représentants de l'État et des fondateurs ainsi agréés. Le procès-verbal de la réunion constate l'approbation des statuts par l'État et les fondateurs agréés.

Article 104 : A compter de la date de la décision qui lui reconnaît cette qualité, tout fondateur agréé jouit des mêmes droits que tout fondateur historique.

SOUS-SECTION 4 — QUALITÉ DE DONATEUR D'UNE FONDATION

Article 105 : Les personnes qui contribuent par le don de leurs biens, droits ou ressources aux activités d'une fondation et qui n'ont pas la qualité de fondateur ont la qualité de donateurs.

Article 106 : Les personnes qui accomplissent un travail bénévole au profit d'une fondation n'ont pas la qualité de donateurs.

SECTION 2 — ORGANISATION - FONCTIONNEMENT DES FONDATIONS

Article 107 : L'organisation et le fonctionnement des fondations sont déterminés par leurs statuts sous réserve des dispositions de la présente loi.

Article 108 : L'organisation de toute fondation comprend au moins un conseil d'administration et un organe de gestion.

SOUS-SECTION 1 — ORGANES D'ADMINISTRATION ET DE GESTION DES FONDATIONS

PARAGRAPHE 1 — CONSEIL D'ADMINISTRATION

Article 109 : Toute fondation est administrée par un conseil d'administration. Le conseil d'administration définit les orientations stratégiques de la fondation. Il a les pouvoirs les plus étendus pour prendre toutes décisions dans l'intérêt de la fondation et, notamment décide des actions en justice, vote le budget, approuve les comptes et décide des emprunts.

Article 110 : Le conseil d'administration de toute fondation comprend au moins deux collèges dont un collège représentant les fondateurs historiques, un collège représentant les fondateurs agréés ou, à défaut, un collège de personnalités qualifiées dans les domaines d'intervention de la fondation. Le nombre de membres du conseil d'administration est fixé par les statuts de la fondation. Toutefois, le nombre de membres constituant chaque collège ne peut excéder la majorité absolue des sièges composant le conseil.

Article 111 : Les personnalités qualifiées visées à l'article 110 de la présente loi sont choisies par les fondateurs ou leurs représentants et nommées lors de la création de la fondation. Il en est fait mention dans le procès-verbal constatant la création de la fondation. A défaut, ils sont désignés dans les trente jours de la publication de la déclaration d'existence de la fondation au registre des associations et fondations. Lorsque la fondation est créée par l'État, comme unique fondateur, les personnalités qualifiées visées à l'article 110 de la présente loi sont nommées dans le décret de création. En cas d'admission de fondateurs agréés, les personnalités qualifiées sont nommées conformément aux statuts révisés et adoptés par les fondateurs historiques et les fondateurs agréés.

Article 112 : Les statuts précisent les conditions de nomination et de renouvellement des membres du conseil d'administration.

Article 113 : Les membres du conseil d'administration exercent leurs fonctions à titre gratuit. Ils peuvent toutefois bénéficier de frais de sujétion en raison de leurs fonctions.

PARAGRAPHE 2 — ORGANE DE GESTION DES FONDATIONS

Article 114 : Toute fondation dispose d'un organe de gestion dont la dénomination est fixée par les statuts de la fondation.

Article 115 : L'organe de gestion est chargé de la gestion quotidienne des affaires de la fondation. Il met en œuvre les orientations décidées par le conseil d'administration. Il représente la fondation dans les actes de la vie civile.

SOUS-SECTION 2 — FONCTIONNEMENT DES FONDATIONS

Article 116 : Les règles de fonctionnement des fondations sont précisées dans leurs statuts dans le respect de la transparence et des objectifs de la fondation concernée.

SECTION 3 — RESSOURCES DES FONDATIONS

Article 117 : Les ressources des fondations peuvent provenir :
- des versements effectués ou des biens affectés par son fondateur ou ses fondateurs ;
- des subventions de l'État, des collectivités territoriales ou de leurs établissements publics ;
- des revenus des activités de la fondation ;
- des dons et legs.

Article 118 : Les biens, droits ou ressources affectés à la création d'une fondation constituent sa dotation initiale et peuvent être libérés en une ou plusieurs fractions dans les conditions prévues par les statuts sur une période qui ne peut excéder trois ans.

Article 119 : Un legs peut être fait au profit d'une fondation qui n'existe pas au jour de l'ouverture d'une succession. Le legs est rétroactivement acquis à la fondation à compter du jour de l'ouverture de la succession à condition de l'accomplissement des formalités de déclaration d'existence au registre des associations et fondations prévues par la présente loi.

SECTION 4 — DISSOLUTION DES FONDATIONS

Article 120 : Toute fondation peut être dissoute conformément à ses statuts ou par décision de justice.

Article 121 : Lorsque la dissolution de la fondation est prononcée conformément à ses statuts, il est nommé, conformément auxdits statuts, un ou plusieurs liquidateurs, chargés de procéder à la liquidation des biens de la fondation et auxquels sont conférés les pouvoirs nécessaires pour mener à bien cette mission. Lorsque la dissolution de la fondation est prononcée par décision de justice, la juridiction saisie nomme un ou plusieurs liquidateurs, chargés de procéder à la liquidation des biens de la fondation.

Article 122 : L'actif net issu de la liquidation d'une fondation est attribué à une ou plusieurs entités poursuivant une finalité analogue. A défaut pour les organes de la fondation d'en avoir délibéré, l'actif net est acquis à l'État.

--- CHAPITRE II : RÈGLES PARTICULIÈRES APPLICABLES AUX FONDATIONS NOMMÉES ---

Article 123 : Sont des fondations nommées :
- la fondation reconnue d'utilité publique ;
- la fondation d'entreprise ;
- la fondation étrangère.

SECTION 1 — FONDATION RECONNUE D'UTILITÉ PUBLIQUE

SOUS-SECTION 1 — DÉCISION DE RECONNAISSANCE DE L'UTILITÉ PUBLIQUE

Article 124 : Les fondations sont reconnues d'utilité publique dans les conditions prévues pour les associations aux articles 74 à 77 de la présente loi. Toutefois, les fondations créées par l'État, comme unique fondateur, ainsi que celles réunissant l'État et d'autres fondateurs, bénéficient de plein droit de la reconnaissance d'utilité publique.

Article 125 : La reconnaissance d'utilité publique peut être accordée ou retirée aux fondations par décret pris en Conseil des ministres. Cette décision peut faire l'objet d'un recours conformément aux dispositions de l'article 86 ci-dessus.

Article 126 : La décision de reconnaissance d'utilité publique est d'office caduque si les statuts de la fondation n'ont pas été modifiés conformément au premier alinéa de l'article 132 de la présente loi et publiés dans le délai de trente jours conformément aux dispositions du premier alinéa de l'article 93 de la présente loi.

SOUS-SECTION 2 — DROITS - OBLIGATIONS LIÉS À LA RECONNAISSANCE D'UTILITÉ PUBLIQUE

Article 127 : Sur tous les actes et documents émanant de la fondation reconnue d'utilité publique et destinés aux tiers, sa dénomination doit être précédée ou suivie immédiatement en caractères lisibles des mots « fondation reconnue d'utilité publique », de l'adresse de son siège et de la mention de sa déclaration d'existence au registre des associations et fondations.

Article 128 : Outre les avantages qui peuvent être accordés par l'État à toute fondation, la fondation reconnue d'utilité publique peut bénéficier d'avantages spécifiques déterminés dans le cadre de l'accord de siège ou d'autres accords conclus avec l'État. Ces avantages peuvent comprendre notamment des exonérations de droits fiscaux et douaniers.

Article 129 : Toute fondation reconnue d'utilité publique dispose d'un commissaire aux comptes qui exécute sa mission telle que prescrite par les textes en vigueur.

Article 130 : Une fondation reconnue d'utilité publique peut recevoir et détenir des parts sociales ou des actions d'une société ayant une activité industrielle ou commerciale pourvu que les fruits soient entièrement destinés à l'accomplissement d'une œuvre d'intérêt général.

Article 131 : La mauvaise gestion de ressources publiques, des fonds du public ou de financements garantis par l'État ou par une collectivité territoriale, est considérée comme un manquement grave aux obligations d'une fondation reconnue d'utilité publique. Le manquement grave prévu à l'alinéa premier du présent article est puni d'un emprisonnement de six mois à un an et d'une amende d'un million (1 000 000) de francs CFA ou de l'une de ces deux peines seulement, sans préjudice de qualifications pénales plus graves.

SOUS-SECTION 3 — CONSEIL D'ADMINISTRATION

Article 132 : Nonobstant les dispositions de l'article 110 de la présente loi, dans toute fondation reconnue d'utilité publique qui n'est pas étrangère, le nombre de membres du collège des fondateurs ne doit pas atteindre la majorité absolue du nombre de sièges composant le conseil d'administration.

Article 133 : Dans les fondations qui ne sont pas créées par l'État comme fondateur unique, un représentant de l'État siège de plein droit, avec voix délibérative, au sein du conseil d'administration dès lors qu'elles sont reconnues d'utilité publique. Les statuts de la fondation sont modifiés en conséquence et publiés par les organes compétents de la fondation, dans les trente jours de la décision de reconnaissance d'utilité publique. Le représentant de l'État est désigné au titre des personnalités qualifiées dans les domaines d'intervention de la fondation. La représentation de l'État peut être assurée par une personne morale de droit public. Celle-ci désigne la personne physique qui la représente.

Article 134 : La désignation de la personne physique ou morale qui représente l'État est faite en Conseil des ministres. Cette désignation intervient dans les trente jours à compter de la publication des statuts modifiés de la fondation au registre des associations et fondations.

SECTION 2 — FONDATION D'ENTREPRISE

SOUS-SECTION 1 — CRÉATION DE LA FONDATION D'ENTREPRISE

Article 135 : Les personnes morales de droit privé ayant pour objet une activité commerciale ou industrielle peuvent, seule ou entre elles ou avec l'État ou d'autres personnes morales de droit public, créer une fondation d'entreprise en vue de la réalisation d'une œuvre d'intérêt général.

Article 136 : Lorsqu'elle est créée avec la participation de l'État ou d'une autre personne morale de droit public, la fondation d'entreprise peut avoir notamment pour objet la création et/ou la gestion d'une entité commune ou la réalisation d'un programme d'actions pluriannuel déterminé par les statuts. Dans ce cas, à l'exception des personnes morales de droit public, les biens ou sommes que chaque fondateur s'engage à affecter à la fondation d'entreprise sont garantis par une caution bancaire.

Article 137 : La fondation d'entreprise est créée pour une durée déterminée librement fixée par ses statuts.

Article 138 : Tout fondateur est tenu de verser intégralement les sommes qu'il s'est engagé à payer ou de remettre les biens qu'il s'est engagé à donner, même lorsqu'il décide de cesser sa participation à l'action de la fondation. Avant le terme de la durée prévue de la fondation, les fondateurs ou certains d'entre eux seulement, peuvent décider de sa prorogation pour une durée qu'ils déterminent en vue de la poursuite des objectifs de la fondation.

Article 139 : Sur tous les actes et documents émanant de la fondation et destinés aux tiers, sa dénomination doit être précédée ou suivie immédiatement en caractères lisibles des mots « fondation d'entreprise », le cas échéant, complétés par la mention « reconnue d'utilité publique », de l'adresse de son siège et de l'indication de sa déclaration d'existence au registre des associations et fondations.

Article 140 : Toute fondation d'entreprise peut solliciter et obtenir la reconnaissance d'utilité publique. La reconnaissance d'utilité publique est accordée dans les mêmes conditions et, sous réserve des dispositions de la présente section relative à la fondation d'entreprise, comporte les mêmes obligations que celles de toute fondation non étrangère.

SOUS-SECTION 2 — CONSEIL D'ADMINISTRATION

Article 141 : Nonobstant les dispositions de l'article 110 de la présente loi, le Conseil d'administration de toute fondation d'entreprise qui n'est pas reconnue d'utilité publique ou qui n'est pas étrangère est composé pour les deux tiers au plus, des fondateurs ou de leurs représentants et pour un tiers au moins, de personnalités qualifiées dans ses domaines d'intervention. Le conseil d'administration d'une fondation d'entreprise non reconnue d'utilité publique, autre que celle à laquelle prend part l'État ou toute autre personne morale de droit public, peut comprendre, pour les deux tiers visés au premier alinéa du présent article, des fondateurs ou de leurs représentants et des représentants du personnel des personnes morales fondatrices.

SOUS-SECTION 3 — RESSOURCES

Article 142 : Nonobstant les dispositions de l'article 117 de la présente loi, toute fondation d'entreprise non reconnue d'utilité publique, autre que celle à laquelle prend part l'État ou une autre personne morale de droit public, ne peut recevoir ni des dons ni de legs, à l'exception de ceux effectués par les salariés, mandataires sociaux, sociétaires, adhérents ou actionnaires des personnes morales fondatrices.

Article 143 : Toute fondation d'entreprise dispose d'un commissaire aux comptes qui exécute sa mission telle que prescrite par les textes en vigueur.

SECTION 3 — RÈGLES PARTICULIÈRES APPLICABLES AUX FONDATIONS ÉTRANGÈRES

Article 144 : Toute fondation étrangère qui désire exercer des activités en République du Bénin est préalablement autorisée par l'autorité compétente chargée de la tenue du registre des associations et fondations, dans les conditions fixées par décret pris en Conseil des ministres.

Article 145 : Toute fondation étrangère autorisée jouit de la personnalité juridique en République du Bénin. La fondation étrangère autorisée jouit des droits et est tenue des obligations prévues par la présente loi pour les fondations sans qu'il n'y ait lieu à la modification de ses statuts en ce qui concerne ses organes de décision et ses règles de fonctionnement. La fondation étrangère autorisée peut être reconnue d'utilité publique sous les conditions prévues à l'article 75 de la présente loi. La reconnaissance d'utilité publique est accordée par décret pris en Conseil des ministres.

Article 146 : Toute fondation étrangère reconnue d'utilité publique conclut de plein droit un accord de siège avec l'État qui lui confère les privilèges consulaires et/ou diplomatiques pendant la durée de validité de l'accord. Les conditions, la procédure de conclusion ainsi que les avantages conférés dans le cadre des accords de siège sont précisés par décret pris en Conseil des ministres.

========================================
TITRE IV
DISPOSITIONS TRANSITOIRES ET FINALES COMMUNES AUX ASSOCIATIONS ET AUX FONDATIONS
========================================

Article 147 : La présente loi est applicable aux associations, aux regroupements d'associations et aux fondations qui sont constitués ou qui exercent leurs activités sur le territoire de la République du Bénin à compter de son entrée en vigueur.

Article 148 : Les associations, regroupements d'associations et fondations constitués antérieurement à l'entrée en vigueur de la présente loi sont soumis à ses dispositions. Ils sont tenus de mettre leurs statuts en harmonie avec les dispositions de la présente loi et ses textes d'application dans les délais fixés par lesdits textes d'application. Sous réserve de cette mise en harmonie, ils conservent leur personnalité juridique acquise conformément à la loi en vigueur au moment de leur constitution. Les fondations constituées sous forme d'association sous le régime de la législation antérieure ne peuvent conserver l'appellation fondation que sous réserve de leur conformité aux dispositions de la présente loi. Les associations, regroupements d'associations et fondations qui ne se conforment pas aux dispositions de la présente loi par la mise en harmonie de leurs statuts sont de plein droit dissous après l'expiration d'un délai de neuf mois à compter de son entrée en vigueur.

Article 149 : Les textes d'application de la présente loi sont pris dans un délai de trois mois, à compter de la date de son entrée en vigueur. La présente loi abroge les dispositions de la loi du 1er juillet 1901 relative au contrat d'association ainsi que toutes autres dispositions antérieures contraires.

Article 150 : La présente loi sera exécutée comme Loi de l'État.

Fait à Cotonou, le 22 juillet 2025.
Par le Président de la République, Chef de l'État, Chef du Gouvernement,
Patrice TALON
Contresigné par :
- Le Garde des Sceaux, Ministre de la Justice et de la Législation : Yvon DETCHENOU
- Le Ministre de l'Intérieur et de la Sécurité publique : Alassane SEIDOU


FISCALITE DES ONG, ASSOCIATIONS ET FONDATIONS AU BENIN
(Source : Livre "Comptabilite des ONG, Associations et Fondations" — Odilon A. MAFON)

=== IMPOTS A PAYER ===

1. IMPOT SUR LE REVENU FONCIER (IRF) — Article 102 CGI :
Les associations, ONG et fondations qui mettent en location des immeubles
sont soumises a l'IRF.
"Sont considerees comme personnes physiques assimilees : les associations,
les organisations non gouvernementales, lorsqu'ils sont titulaires uniquement
de revenus fonciers."

2. VERSEMENT PATRONAL SUR SALAIRES (VPS) — Articles 191-192 CGI :
Les associations et ONG qui paient des salaires sont assujetties au VPS.
Exception : les associations et ONG peuvent beneficier d'une exoneration
du VPS en fonction des accords avec l'Etat.
Les associations ne beneficiant pas d'une exoneration expresse ne sont
pas concernees.

3. TAXE FONCIERE UNIQUE (TFU) — Articles 151-152 CGI :
Si l'association, l'ONG ou la fondation est proprietaire d'un immeuble
(bati ou non), elle doit payer la TFU, sauf derogation expresse.

4. TAXE SUR LES VEHICULES A MOTEUR (TVM) — Article 166 CGI :
Les vehicules a moteur d'au moins 3 roues immatricules et utilises pour
le transport sont soumis a la TVM annuelle.

=== EXONERATIONS ===

1. IMPOT SUR LES SOCIETES (IS) — Article 4-9 CGI :
Les associations et organismes sans but lucratif legalement constitues
et dont la gestion est desinteressee sont EXONERES de l'IS.
Conditions de gestion desinteressee :
a) Gere a titre benevole — remuneration possible si transparence financiere
   et remuneration mensuelle ne depassant pas 10 fois le SMIG.
b) Depot rapport d'activite au plus tard le 30 avril de chaque annee.

2. PLUS-VALUES — Article 19 CGI :
Les associations a but non lucratif et les fondations ne peuvent pas
beneficier du regime de faveur sur les reevaluations.

3. REVENUS DES CREANCES — Article 80 CGI :
Sont exoneres de l'IRCM : les parts d'interet, emprunts ou obligations
des societes cooperatives agricoles et associations agricoles.

4. TVA — Article 229 CGI :
Sont exoneres de la TVA :
"Les services rendus benevolement ou a un prix egal ou inferieur au prix
de revient par les associations et organismes vises au paragraphe 9 de
l'article 4 du CGI et les etablissements d'utilite publique."

5. TAXE SUR LES VEHICULES A MOTEUR (TVM) — Article 167 CGI :
Sont exempts de TVM :
- Les vehicules immatricules au nom de l'Etat beninois
- Les vehicules immatricules au nom du corps diplomatique et
  organisations internationales
- Les vehicules des ONG internationales ayant signe un accord de
  siege avec la Republique du Benin

=== RETENUES A LA SOURCE ===
Les associations et ONG ne paient pas directement ces impots mais doivent
les retenir et les reverser a l'Etat :

1. IRCM — Article 69 CGI :
Soumis a l'IRCM : les revenus verses aux membres des conseils
d'administration des organismes et associations a but non lucratif.

2. ITS (Impot sur les Traitements et Salaires) — Article 128 CGI :
L'impot est preleve a la source au moment du paiement des salaires
par tout employeur, y compris les associations.

3. AIB (Acompte d'Impot assis sur le Benefice) — Articles 130-134 CGI :
Les associations et organismes a but non lucratif nationaux et
internationaux doivent retenir l'AIB sur tous les paiements faits aux
fournisseurs de travaux, biens et prestataires de services.
L'AIB est retenu a la source par les associations et reverse a la DGI.

=== AUTRES OBLIGATIONS ===

1. ENREGISTREMENT DES ACTES — Article 327 CGI :
Sont enregistres gratuitement les actes des associations dont les
recettes annuelles sont constituees a 80% au moins de fonds publics.

2. OBLIGATIONS SOCIALES :
Les associations employeurs doivent :
- Declarer les salaires a la CNSS
- Payer les cotisations sociales patronales
- Respecter les obligations du Code du Travail

3. DECLARATION ANNUELLE DES SOMMES VERSEES AUX TIERS :
Les associations doivent declarer annuellement les sommes versees
aux tiers (prestataires, fournisseurs).

4. ATTESTATION DE REGULARITE FISCALE :
Les associations doivent disposer d'une attestation de regularite
fiscale pour certaines operations.

5. DEPOT DU RAPPORT D'ACTIVITE :
Obligation de deposer le rapport d'activite (moral et financier) au plus
tard le 30 avril de chaque annee aupres des services fiscaux.

=== TABLEAU RECAPITULATIF ===
| Impot/Taxe | Association EBNL | Condition |
| IS | EXONERE | Gestion desinteressee + depot rapport |
| TVA | EXONERE | Services benevoles ou prix <= prix de revient |
| IRF | DU | Si location d'immeuble |
| TFU | DUE | Si proprietaire d'immeuble |
| TVM | DUE | Si vehicule immatricule (sauf accord siege) |
| VPS | EXONERE possible | Selon accord avec l'Etat |
| ITS | A RETENIR | Sur salaires payes |
| AIB | A RETENIR | Sur paiements fournisseurs |
| IRCM | A RETENIR | Sur revenus administrateurs |



ECRITURES TYPES VALIDEES — EGLISE :

COMPTES DE REFERENCE :
- Caisse : 5711 | Banque : 5211 | Virement de fonds : 5850
- Fideles/Adherents : 4111 | Fournisseur : 4011
- Personnel du : 422 | Securite sociale : 4318
- Fournisseurs investissement immo corporelle : 4812
- Fournisseurs investissement immo incorporelle : 4811

=== RECETTES ===

E01 — Collecte dime/offrande en especes :
  Tresorerie (1 ecriture) :
    Debit  5711 Caisse / Credit 7044 Dimes recues
    Libelle : S/Encaissement des dimes
  Engagement (2 ecritures) :
    Debit  4111 Fideles / Credit 7044 Dimes recues — Libelle : S/Engagement
    Debit  5711 Caisse / Credit 4111 Fideles — Libelle : S/Encaissement des quetes

E02 — Collecte dime/offrande par banque :
  Tresorerie (1 ecriture) :
    Debit  5211 Banque / Credit 7044 Dimes recues
    Libelle : S/Encaissement des dimes
  Engagement (2 ecritures) :
    Debit  4111 Fideles / Credit 7044 Dimes recues — Libelle : S/Engagement
    Debit  5211 Banque / Credit 4111 Fideles — Libelle : S/Encaissement

E03 — Versement collectes en banque :
  Tresorerie ET Engagement (2 ecritures identiques) :
    Debit  5211 Banque / Credit 5850 Virement de fonds — Libelle : S/Bordereau de versement
    Debit  5850 Virement de fonds / Credit 5711 Caisse — Libelle : S/Avis de credit

E04 — Cotisation annuelle fidelee par banque :
  Tresorerie (1 ecriture) :
    Debit  5211 Banque / Credit 7011 Cotisations
    Libelle : S/Encaissement de cotisation
  Engagement (2 ecritures) :
    Debit  4111 Adherents / Credit 7011 Cotisations — Libelle : S/Engagement
    Debit  5211 Banque / Credit 4111 Adherents — Libelle : S/Encaissement de cotisation

E04b — Cotisation annuelle fidele en especes :
  Tresorerie (1 ecriture) :
    Debit  5711 Caisse / Credit 7011 Cotisations
    Libelle : S/Encaissement de cotisation
  Engagement (2 ecritures) :
    Debit  4111 Adherents / Credit 7011 Cotisations — Libelle : S/Engagement
    Debit  5711 Caisse / Credit 4111 Adherents — Libelle : S/Encaissement de cotisation

E05 — Recette funerailles/bapteme/mariage en especes :
  Tresorerie (1 ecriture) :
    Debit  5711 Caisse / Credit 7061 Revenus des manifestations
    Libelle : S/Encaissement
  Engagement (2 ecritures) :
    Debit  4111 Adherents / Credit 7061 Revenus des manifestations — Libelle : S/Engagement
    Debit  5711 Caisse / Credit 4111 Adherents — Libelle : S/Encaissement

E05b — Recette funerailles/bapteme/mariage par banque :
  Tresorerie (1 ecriture) :
    Debit  5211 Banque / Credit 7061 Revenus des manifestations
    Libelle : S/Encaissement
  Engagement (2 ecritures) :
    Debit  4111 Adherents / Credit 7061 Revenus des manifestations — Libelle : S/Engagement
    Debit  5211 Banque / Credit 4111 Adherents — Libelle : S/Encaissement

E07 — Vente de Bibles en especes :
  Tresorerie (1 ecriture) :
    Debit  5711 Caisse / Credit 7081 Ventes de dons en nature
    Libelle : S/Encaissement de la vente
  Engagement (2 ecritures) :
    Debit  4111 Adherents / Credit 7081 Ventes de dons en nature — Libelle : S/Engagement
    Debit  5711 Caisse / Credit 4111 Adherents — Libelle : S/Encaissement de la vente

E07b — Vente livres religieux produits par eglise en especes :
  Tresorerie (1 ecriture) :
    Debit  5711 Caisse / Credit 7053 Ventes de produits finis
    Libelle : S/Encaissement de la vente
  Engagement (2 ecritures) :
    Debit  4111 Adherents / Credit 7053 Ventes de produits finis — Libelle : S/Engagement
    Debit  5711 Caisse / Credit 4111 Adherents — Libelle : S/Encaissement de la vente

E07c — Vente livres religieux achetes par eglise en especes :
  Tresorerie (1 ecriture) :
    Debit  5711 Caisse / Credit 7051 Ventes de marchandises
    Libelle : S/Encaissement de la vente
  Engagement (2 ecritures) :
    Debit  4111 Adherents / Credit 7051 Ventes de marchandises — Libelle : S/Engagement
    Debit  5711 Caisse / Credit 4111 Adherents — Libelle : S/Encaissement de la vente

E08 — Loyer encaisse bien immobilier eglise en especes :
  Tresorerie (1 ecriture) :
    Debit  5711 Caisse / Credit 7071 Produits accessoires
    Libelle : S/Encaissement
  Engagement (2 ecritures) :
    Debit  4111 Adherents / Credit 7071 Produits accessoires — Libelle : S/Engagement
    Debit  5711 Caisse / Credit 4111 Adherents — Libelle : S/Encaissement

E09 — Interets bancaires credites DAT :
  Tresorerie ET Engagement (1 ecriture) :
    Debit  5211 Banque / Credit 7747 Revenus des DAT
    Libelle : S/Encaissement interets

=== CHARGES ET DEPENSES ===

E11 — Achat cierges/encens/accessoires liturgiques par banque :
  Tresorerie (1 ecriture) :
    Debit  6047 Achats fournitures liturgiques / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6047 Achats fournitures liturgiques / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5211 Banque — Libelle : S/Paiement

E12 — Reglement fournisseur fournitures en caisse :
  Tresorerie ET Engagement (1 ecriture) :
    Debit  4011 Fournisseur / Credit 5711 Caisse
    Libelle : S/Paiement

E13 — Paiement loyer salle de culte par banque :
  Tresorerie (1 ecriture) :
    Debit  6222 Loyers / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6222 Loyers / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5211 Banque — Libelle : S/Paiement

E14 — Paiement prime assurance par banque :
  Tresorerie (1 ecriture) :
    Debit  6252 Assurances materiel transport / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6252 Assurances materiel transport / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5211 Banque — Libelle : S/Paiement

E15 — Versement salaire pasteur/employe en caisse :
  Tresorerie (1 ecriture) :
    Debit  6611 Remunerations / Credit 5711 Caisse
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6611 Remunerations / Credit 422 Personnel du — Libelle : S/Engagement
    Debit  422 Personnel du / Credit 5711 Caisse — Libelle : S/Paiement

E17 — Cotisations sociales patronales par banque :
  Tresorerie (1 ecriture) :
    Debit  6641 Charges sociales / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6641 Charges sociales / Credit 4318 Securite sociale — Libelle : S/Engagement
    Debit  4318 Securite sociale / Credit 5211 Banque — Libelle : S/Paiement

E19 — Honoraires predicateur invite par banque :
  Tresorerie (1 ecriture) :
    Debit  6057 Honoraires / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6057 Honoraires / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5211 Banque — Libelle : S/Paiement

E20 — Frais deplacement mission evangelisation en caisse :
  Tresorerie (1 ecriture) :
    Debit  6181 Voyages et Deplacements / Credit 5711 Caisse
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6181 Voyages et Deplacements / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5711 Caisse — Libelle : S/Paiement

E20b — Facture electricite par banque :
  Tresorerie (1 ecriture) :
    Debit  6052 Electricite / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6052 Electricite / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5211 Banque — Libelle : S/Paiement

E21 — Facture eau en caisse :
  Tresorerie (1 ecriture) :
    Debit  6051 Eau / Credit 5711 Caisse
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6051 Eau / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5711 Caisse — Libelle : S/Paiement

E21b — Facture internet en caisse :
  Tresorerie (1 ecriture) :
    Debit  6288 Autres frais telecommunications / Credit 5711 Caisse
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6288 Autres frais telecommunications / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5711 Caisse — Libelle : S/Paiement

E21c — Facture essence/gazoil en caisse :
  Tresorerie (1 ecriture) :
    Debit  6053 Autres energies / Credit 5711 Caisse
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6053 Autres energies / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5711 Caisse — Libelle : S/Paiement

E22 — Frais bancaires commission :
  Tresorerie ET Engagement (1 ecriture) :
    Debit  6318 Services bancaires / Credit 5211 Banque
    Libelle : S/Paiement

E23 — Versement subvention a une association par banque :
  Tresorerie (1 ecriture) :
    Debit  6520 Subventions versees / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6520 Subventions versees / Credit 4712 Crediteurs divers — Libelle : S/Engagement
    Debit  4712 Crediteurs divers / Credit 5211 Banque — Libelle : S/Paiement

E24 — Achat vivres repas communautaire par banque :
  Tresorerie (1 ecriture) :
    Debit  6046 Fournitures de magasin / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  6046 Fournitures de magasin / Credit 4011 Fournisseur — Libelle : S/Engagement
    Debit  4011 Fournisseur / Credit 5211 Banque — Libelle : S/Paiement

=== OPERATIONS EN CAPITAL ===

C01 — Acquisition bien immobilier edifice par banque :
  Tresorerie (1 ecriture) :
    Debit  2317 Edifices religieux / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  2317 Edifices religieux / Credit 4812 Fournisseurs investissement — Libelle : S/Engagement
    Debit  4812 Fournisseurs investissement / Credit 5211 Banque — Libelle : S/Paiement

C01b — Conception logiciel par banque :
  Tresorerie (1 ecriture) :
    Debit  2131 Logiciels / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  2131 Logiciels / Credit 4811 Fournisseurs investissement immo incorporelle — Libelle : S/Engagement
    Debit  4811 Fournisseurs investissement / Credit 5211 Banque — Libelle : S/Paiement

C02 — Emprunt bancaire pour construction :
  Tresorerie ET Engagement (1 ecriture) :
    Debit  5211 Banque / Credit 182 Emprunt bancaire
    Libelle : S/Encaissement emprunt

C04 — Achat mobilier religieux par banque :
  Tresorerie (1 ecriture) :
    Debit  2443 Materiel et mobilier religieux / Credit 5211 Banque
    Libelle : S/Paiement
  Engagement (2 ecritures) :
    Debit  2443 Materiel et mobilier religieux / Credit 4812 Fournisseurs investissement — Libelle : S/Engagement
    Debit  4812 Fournisseurs investissement / Credit 5211 Banque — Libelle : S/Paiement

C05 — Avance sur salaire accordee en caisse :
  Tresorerie ET Engagement (1 ecriture) :
    Debit  422 Personnel du / Credit 5711 Caisse
    Libelle : S/Paiement avance


=== ECRITURES COMPTABLES GENERALES DES ASSOCIATIONS ===

A. COTISATIONS DES MEMBRES :
Appel de cotisation :
  Debit  411 Adherents / Credit 701 Cotisations des adherents
Recouvrement :
  Debit  5711 Caisse ou 5211 Banque / Credit 411 Adherents
Creance douteuse :
  Debit  4161 Adherents cotisations douteuses / Credit 411 Adherents
  Debit  6594 Charges depreciations creances / Credit 4912 Depreciations creances douteuses

B. DROIT D'ENTREE (nouveau membre) :
Sans precision (par nature) :
  Debit  411 Adherents / Credit 103 Droit d'entree
Si objectif de financer l'entite :
  Debit  411 Adherents / Credit 1851 Depots recus
S'il represente un produit :
  Debit  411 Adherents / Credit 701 Cotisations des adherents
Paiement :
  Debit  5211 Banque ou 5711 Caisse / Credit 411 Adherents

C. APPORTS :
Souscription des apports :
  Debit  45 Fondateurs apporteurs / Credit 101 Dotation non consomptible sans droit reprise
  Debit  45 Fondateurs apporteurs / Credit 102 Dotation non consomptible avec droit reprise
  Debit  45 Fondateurs apporteurs / Credit 1041 Dotation consomptible
Liberation des apports :
  Debit  2 Immobilisations / Credit 45 Fondateurs apporteurs
  Debit  5 Tresorerie / Credit 45 Fondateurs apporteurs

D. SUBVENTIONS D'INVESTISSEMENT :
Notification de la subvention :
  Debit  4731 Subventions a recevoir / Credit 14 Subventions d'investissement
Reception des fonds :
  Debit  5211 Banque / Credit 4731 Subventions a recevoir
Acquisition de l'immobilisation :
  Debit  2 Immobilisations / Credit 4812 Fournisseurs d'investissement
  Debit  4812 Fournisseurs d'investissement / Credit 5211 Banque
A la cloture — Amortissement :
  Debit  6813 Dotations amortissements / Credit 284 Amortissement materiel
Reprise subvention :
  Debit  14 Subventions d'investissement / Credit 799 Reprises subventions investissement

E. FONDS AFFECTES A UN PROJET :
Reception des fonds :
  Debit  5 Tresorerie / Credit 165 Fonds affectes a un projet specifique
Au fur et a mesure des consommations :
  Debit  165 Fonds affectes / Credit 7925 Reprises fonds affectes projet specifique

F. CONTRIBUTIONS VOLONTAIRES EN NATURE (benevoles) :
  Debit  900 Secours en nature / Credit 910 Dons en nature
  Debit  904 Personnel benevole / Credit 914 Benevolat
NB : Valorisation au taux horaire du SMIG (300 FCFA/heure)

G. REEVALUATION DES IMMOBILISATIONS :
  Debit  22 Terrain (ou autre immo) / Credit 106 Ecart de reevaluation

H. RESERVES :
Affectation excedent :
  Debit  131 Resultat net excedent / Credit 11 Reserves
Incorporation aux reserves :
  Debit  11 Reserves / Credit 101 ou 102 Dotation non consomptible
Imputation deficit :
  Debit  11 Reserves / Credit 139 Resultat net deficit

=== ECRITURES SALAIRES ET CHARGES SOCIALES ===

I. CONSTATATION DU SALAIRE :
  Debit  6611 Appointements salaires / Credit 422 Personnel remunerations dues (salaire brut)

J. DEDUCTIONS (retenues sur salaire) :
  Debit  422 Personnel remunerations dues
    Credit 4211 Personnel avances
    Credit 4472 Impots sur salaires ITS
    Credit 4318 Autres cotisations sociales CNSS 3.6%
NB : En mars deduire 1000 FCFA ORTB et 3000 FCFA en juin (compte 4478)

K. VERSEMENT DU SALAIRE NET :
  Debit  422 Personnel remunerations dues / Credit 5 Tresorerie

L. CHARGES PATRONALES :
  Debit  6641 Charges sociales
    Credit 4311 Prestations familiales 9% salaire brut
    Credit 4312 Accidents de travail (taux de risque)
    Credit 4318 Autres cotisations sociales 6.4%

M. VPS (Versement Patronal sur Salaires) :
  Debit  6413 Taxes sur appointements et salaires / Credit 4428 Autres impots et taxes
Reversement le 10 du mois suivant :
  Debit  4428 / Credit 5 Tresorerie

=== ECRITURES IMPOTS ET TAXES ===

N. IRF (Impot sur Revenu Foncier) :
Constatation : Debit 6411 Impots fonciers / Credit 4421 Impots et taxes d'Etat
Paiement au 10 fevrier : Debit 4421 / Credit 5 Tresorerie

O. TFU (Taxe Fonciere Unique) :
Constatation : Debit 6411 / Credit 4422 Impots taxes collectivites publiques
Paiement 10 fevrier (50%) et 30 avril (50%) : Debit 4422 / Credit 5 Tresorerie

P. TVM (Taxe Vehicules a Moteur) :
Constatation : Debit 6464 Vignettes / Credit 4421 Impots et taxes d'Etat
Paiement au 30 avril : Debit 4421 / Credit 5 Tresorerie

Q. AIB (Retenue sur prestations) :
Constatation prestation : Debit 6327 Remunerations autres prestataires / Credit 4011 Fournisseur
Paiement net (3% immatricule / 5% non immatricule / 20% etranger) :
  Debit 4011 Fournisseur / Credit 5 Tresorerie + Credit 4478 Autres impots contributions
Reversement AIB le 10 du mois suivant : Debit 4478 / Credit 5 Tresorerie

R. IRCM (sur remunerations administrateurs) :
Retenue : Credit 4478 Autres impots et contributions
Reversement le 10 du mois suivant : Debit 4478 / Credit 5 Tresorerie

R. LOYER RETENU A LA SOURCE (IRF sur loyer paye) :
Constatation loyer : Debit 6222 Loyers / Credit 4011 Fournisseur
Paiement net (loyer - 12% IRF) :
  Debit 4011 Fournisseur / Credit 5 Tresorerie + Credit 4478 AIB/retenue
Reversement retenue le 10 fevrier : Debit 4478 / Credit 5 Tresorerie

REGLE ABSOLUE SUR LES ECRITURES :
- Remplace X par le montant exact mentionne par l'utilisateur
- Ne changes JAMAIS les numeros de comptes
- Si l'operation ne correspond a aucune ecriture ci-dessus, dis :
  "Cette operation n'est pas encore dans ma base d'ecritures.
   Veuillez contacter un expert-comptable."

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
