# SRI - Système de Recherche d'Information (Version Web)

Version web de l'application SRI avec une interface HTML/CSS moderne.

## Installation

1. **Installez les dépendances** :
```bash
pip install -r requirements.txt
```

2. **Lancez le serveur** :
```bash
python app.py
```

3. **Ouvrez votre navigateur** :
Accédez à `http://localhost:5000`

## Structure du projet

```
SRI/
├── app.py                    # Application Flask (serveur)
├── requirements.txt          # Dépendances Python
├── templates/
│   └── index.html           # Page HTML principale
├── static/
│   ├── style.css            # Feuille de style CSS
│   └── script.js            # Scripts JavaScript
└── documents/               # Dossier des documents (généré automatiquement)
    ├── introduction_ia.txt
    ├── python_programming.txt
    ├── machine_learning.txt
    ├── bases_de_donnees.txt
    ├── reseaux_informatiques.txt
    ├── algorithmes_tri.txt
    ├── securite_informatique.txt
    └── developpement_web.txt
```

## Fonctionnalités

- 🔍 **Recherche multi-modèles** : 8 algorithmes de recherche différents
  - Cosinus
  - Booléen
  - Booléen étendu (défaut)
  - Lukasiewicz
  - Kraft
  - Jaccard
  - Dice
  - Euclidienne

- 📄 **Support multi-formats** : TXT et PDF

- 🎨 **Interface moderne** : Design responsive et convivial

- ⚡ **Performance** : Affichage rapide des résultats

- 🔤 **Traitement du texte** : Tokenization, stemming, suppression des mots vides

## Modèles de recherche

### Booléen étendu (défaut)
Trouve les documents contenant les termes de recherche. Tous les termes doivent être présents pour obtenir un score maximal.

### Cosinus
Utilise la similarité cosinus entre le vecteur de requête et les vecteurs de documents (TF-IDF).

### Booléen classique
Recherche booléenne simple basée sur la présence des termes.

### Lukasiewicz
Utilise la logique multivaluée de Łukasiewicz pour le calcul de similarité.

### Kraft
Calcule la pertinence basée sur le nombre de termes correspondants au carré.

### Jaccard
Mesure l'intersection sur l'union entre l'ensemble de requête et l'ensemble de documents.

### Dice
Coefficient de Dice pour la similarité entre deux ensembles.

### Euclidienne
Distance euclidienne normalisée pour la similarité.

## Utilisation

1. Entrez une requête dans la barre de recherche
2. Choisissez un modèle de recherche
3. Appuyez sur Entrée ou cliquez sur le bouton 🔍
4. Cliquez sur un résultat pour voir le document complet

## Personnalisation

Pour ajouter vos propres documents :
1. Placez vos fichiers `.txt` ou `.pdf` dans le dossier `documents/`
2. Redémarrez l'application

## Notes de compatibilité

- Python 3.7+
- Navigateur moderne (Chrome, Firefox, Safari, Edge)
- Testé sur Windows, macOS, Linux
