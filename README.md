# RAG_LIRIS
Ce répertoire contient le code d'un projet de recherche effectué dans le cours Theory and Practical Applications of Large Language Models du Master2 - IA de l'université Claude Bernard - Lyon 1.


## Objectif

L'objectif de ce projet est de créer un agent conversationnel capable de répondre à des questions en rapport avec le LIRIS (Laboratoire d'Informatique en Image et Systèmes d'information) de l'université.

Pour cela, nous mettons en place une génération augmentée de récupération (RAG) pour récupérer les informations pertinentes contenues dans notre base de données.

Par la suite, nous utilisons le modèle Mistral NeMo de NVIDIA pour générer des réponses à partir des informations fournies par le RAG.

## Installation

Les différentes bibliothèques nécessaires pour exécuter le code sont listées dans le fichier `requirements.txt`.

Pour exécuter le code, il est également nécessaire d'installer [Ollama](https://ollama.com/). Comme nous utilisons le modèle Mistral NeMo, il faut aussi lancer la commande suivante pour l'installer (7GB) :
```bash
ollama run mistral-nemo
```

Le démarrage du serveur Web se fait par la commande :
```bash
streamlit run streamlit.py
```

Lors de la première utilisation, les embeddings ne seront pas encore calculés car le fichier est trop lourd pour être stocké sur GitHub. Un bouton *Loading embeddings* permet de les calculer et de les sauvegarder pour les prochaines utilisations. Une fois les embeddings, chargés, il est possible de poser des questions à l'agent conversationnel.

# Video
Cliquez sur l'image pour accéder à notre vidéo de démonstration ou [ici](https://www.youtube.com/watch?v=sGirjQBC5cI)
[![Watch the video](https://img.youtube.com/vi/sGirjQBC5cI/maxresdefault.jpg)](https://www.youtube.com/watch?v=sGirjQBC5cI)

