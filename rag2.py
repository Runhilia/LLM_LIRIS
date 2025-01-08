import json
import torch
import os
import ollama
import numpy as np

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine_similarity
from torch.nn.functional import cosine_similarity as torch_cosine_similarity

# Ouvre un fichier json et renvoie les informations
def parse_file(fichier):
    elements = []
    with open("data/"+fichier, "r", encoding="utf-8") as f:
        elements = json.load(f)
    return elements

# Fonction pour extraire les mots-clés d'un long texte
def extraction_mots_cles(documents, fichier):
    # Si les mots-clés sont déjà extraits, on ne refait pas le travail
    if documents[0].get("mots_cles_description", None) is None:
        keywords_modele = KeyBERT()

        for doc in documents:
            texte = f"{doc.get('title_s', '')} {doc.get('abstract_s', '')}"
            keywords = keywords_modele.extract_keywords(texte, keyphrase_ngram_range=(1, 4), top_n=7) 
            doc['extracted_keywords'] = [kw[0] for kw in keywords]  # Ajouter les mots-clés

        # Une fois les documents modifiés, on les sauvegarde pour pas avoir à exécuter à nouveau la fonction
        with open("./data/"+ fichier + "ExtractedKeywords2.json", "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=4, ensure_ascii=False)
    return documents

# Extraction des informations des documents
def extraction_informations_documents(documents):
    informations = [
        f"{doc.get('title_s', '')} "  # Titre
        f"{', '.join(doc.get('keyword_s', []))} "  # Mots-clés
        f"{', '.join(doc.get('extracted_keywords', []))} "  # Mots-clés extraits
        f"Authors: {', '.join(doc.get('authFullName_s', []))} "  # Auteurs
        f"Date: {doc.get('producedDateY_i', '')} "  # Date de publication
        for doc in documents
    ]
    return informations

# Extraction des informations des équipes
def extraction_informations_equipes(equipes):
    equipes = equipes.get('teams', []).values()
    informations = [
        f"Equipe_name: {equipe.get('name', '')} "  # Nom
        f"Infos: {equipe.get('infos', '')} "  # Infos
        f"Description: {equipe.get('description', '')} "  # Description
        f"Divisions: {', '.join(division.get('division_name', '') for division in equipe.get('divisions', []))} "  # Divisions
        f"Members: {', '.join(member.get('complete_name', '') for member in equipe.get('members', []))} "  # Membres
        for equipe in equipes
    ]
    return informations


###### EMBEDDINGS ######

# Sauvegarde des embeddings
def sauvegarde_embeddings(fichier, embeddings):
    # Crée un dossier pour les embeddings s'il n'existe pas
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # Sauvegarde les embeddings dans un fichier json
    with open(f"embeddings/embeddings_{fichier}", "w", encoding="utf-8") as f:
        json.dump(embeddings.tolist(), f)

# Chargement des embeddings
def chargement_embeddings(fichier):
    # Vérifie si les embeddings sont déjà sauvegardés
    if not os.path.exists(f"embeddings/embeddings_{fichier}"):
        return False
    # Charge les embeddings
    with open(f"embeddings/embeddings_{fichier}", "r", encoding="utf-8") as f:
        return torch.tensor(json.load(f))

# Récupère les embeddings des documents
def get_embeddings(fichier, informations):
    # Vérifie si les embeddings sont déjà sauvegardés
    if (embeddings := chargement_embeddings(fichier)) is not False:
        return embeddings
    
    # Génération des embeddings
    modele = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    embeddings = modele.encode(informations, convert_to_tensor=True)

    # Sauvegarde les embeddings
    sauvegarde_embeddings(fichier, embeddings)
    return embeddings


# Fonction pour extraire les mots-clés d'une query en utilisant un LLM
def get_mots_cles_query(query):
    SYSTEM_PROMPT = """You are an AI assistant that returns keywords from a query given in context.
    You simply need to extract the most relevant words from the query for use in Retrieval Augmented Generation.
    I don't want lots of little keywords, just the most relevant ones.
    Context:
    """
    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(query)
            },
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"]

# Fonction pour trouver les embeddings les plus similaires
def trouve_similaire(query, query_embeddings, embeddings, informations):
    # On met les embeddings et le prompt sur le même device
    embeddings = embeddings.to(query_embeddings.device)
    # On ajoute une dimension batch pour le prompt
    query_embeddings = query_embeddings.unsqueeze(0)
    
    # Recherche sémantique sur les embeddings
    score_similarite = torch_cosine_similarity(query_embeddings, embeddings, dim=1)
    # TF-IDF pour les mots-clés
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(informations)
    tfidf_query = tfidf_vectorizer.transform([query])
    score_tfidf = torch.tensor(tfidf_cosine_similarity(tfidf_query, tfidf_matrix).flatten(), device=query_embeddings.device)

    # Combinaison des scores
    score_combinaison = 0.3 * score_similarite + 0.7 * score_tfidf
    
    # Récupération des indices des documents les plus similaires
    indices = score_combinaison.argsort(descending=True).flatten()
    return [(score_combinaison[i].item(), i) for i in indices]

# Récupération des données et préparation des embeddings
def prepare_embeddings():
    # Fichiers utilisés pour le RAG
    fichiers = ["documentsExtractedKeywords.json", "equipes.json"]
    embeddings = None
    for fichier in fichiers:
        elements = parse_file(fichier) # Ouverture du fichier json et récupération des éléments
    
        if fichier == "documentsExtractedKeywords.json":
            informations = extraction_informations_documents(elements) # Extractions des informations des documents
        else:
            informations = extraction_informations_equipes(elements) # Extractions des informations des équipes
        #embeddings.append(get_embeddings(fichier, informations)) # Création des embeddings
        embeddings = get_embeddings(fichier, informations) # Création des embeddings
    return embeddings

# Génération d'une réponse à partir d'un prompt
def generate_response(prompt_input, embeddings):
    # Extraction des informations des documents et des équipes
    informations_documents = extraction_informations_documents(parse_file("documentsExtractedKeywords.json"))
    informations_equipes = extraction_informations_equipes(parse_file("equipes.json"))
    
    # Récupération de la query et de son embedding
    query = prompt_input
    query_mots_cles = get_mots_cles_query(query)
    modele = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    query_embedding = modele.encode(query_mots_cles, convert_to_tensor=True).to("cuda")
    '''
    # Trouve les documents les plus similaires
    doc_similaires = trouve_similaire(query, query_embedding, embeddings, informations_documents)[:8]
    for score, i in doc_similaires:
        print(f"Score: {score:.4f} - {informations_documents[i]}")
    '''

    # Trouve les équipes qui correspondent le mieux
    equipes_similaires = trouve_similaire(query, query_embedding, embeddings, informations_equipes)[:3]
    for score, i in equipes_similaires:
        print(f"Score: {score:.4f} - {informations_equipes[i]}")

    SYSTEM_PROMPT = """You are an AI assistant who answers user questions based on the documents provided in the context.
    Answer only using the context provided and answer with several sentences about the important information.
    When it comes to documents, use the information provided to go into a little more detail using several sentences.
    If you're not sure, just say you don't know how to answer.
        Context:
    """

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            #    + "\n".join(informations_documents[document] for _, document in doc_similaires)
                + "\n".join(informations_equipes[team] for _, team in equipes_similaires)
            },
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"]