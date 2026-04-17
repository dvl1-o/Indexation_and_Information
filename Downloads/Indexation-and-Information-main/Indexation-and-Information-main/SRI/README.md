# SRI - Système de Recherche d'Information

## Description
Application de recherche documentaire avec interface graphique Tkinter. Supporte 8 modèles de recherche et indexe automatiquement les fichiers `.txt` et `.pdf`.

## Structure
```
SRI/
├── main.py           # Application principale
├── documents/        # Collection de documents (txt ou pdf)
└── README.md
```

## Installation
```bash
pip install nltk PyPDF2
python -c "import nltk; nltk.download('stopwords')"
```

## Exécution
```bash
python main.py
```

## Modèles disponibles
- **Cosinus** : similarité vectorielle TF-IDF
- **Booléen** : correspondance exacte des termes
- **Booléen étendu** : booléen avec score normalisé
- **Lukasiewicz** : logique floue (t-norme)
- **Kraft** : score quadratique pondéré
- **Jaccard** : intersection sur union des termes
- **Dice** : double intersection normalisée
- **Euclidienne** : distance vectorielle inversée

## Ajouter des documents
Placez vos fichiers `.txt` ou `.pdf` dans le dossier `documents/` et relancez l'application.

## Technologies
- Python 3.8+
- Tkinter (interface)
- NLTK (stopwords, stemmer)
- PyPDF2 / pdfminer (lecture PDF)
