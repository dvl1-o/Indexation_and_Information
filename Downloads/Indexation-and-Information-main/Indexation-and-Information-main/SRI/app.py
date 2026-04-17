from flask import Flask, render_template, request, jsonify
import os
import math
import re
import time
import collections

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False


FRENCH_STOPWORDS = {
    "le","la","les","un","une","des","de","du","en","et","est","au","aux",
    "ce","se","sa","son","ses","mon","ma","mes","ton","ta","tes","nous",
    "vous","ils","elles","qui","que","quoi","dont","où","je","tu","il",
    "elle","on","par","sur","sous","dans","avec","pour","sans","entre",
    "vers","chez","après","avant","mais","ou","donc","or","ni","car",
    "plus","moins","très","bien","comme","si","ne","pas","plus","tout",
    "tous","toute","toutes","cette","cet","ces","leur","leurs","même",
    "aussi","alors","ainsi","lors","quand","comment","pourquoi","être",
    "avoir","faire","dit","a","à","y","n","s","l","d","j","m","c",
    "the","of","and","is","in","it","to","that","for","on","are","with",
    "as","at","be","by","from","or","an","this","which","was","but"
}


def get_stopwords():
    if NLTK_AVAILABLE:
        try:
            sw = set(stopwords.words('french')) | set(stopwords.words('english'))
            return sw | FRENCH_STOPWORDS
        except:
            pass
    return FRENCH_STOPWORDS


def get_stemmer():
    if NLTK_AVAILABLE:
        try:
            return SnowballStemmer('french')
        except:
            pass
    return None


def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt':
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            return ""
    elif ext == '.pdf':
        text = ""
        if PYPDF2_AVAILABLE:
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"
                if text.strip():
                    return text
            except:
                pass
        if PDFMINER_AVAILABLE:
            try:
                text = pdfminer_extract(filepath)
                if text and text.strip():
                    return text
            except:
                pass
        return text
    return ""


def tokenize(text, stop_words, stemmer=None):
    text = text.lower()
    text = re.sub(r'[^a-zàâäéèêëîïôùûüç\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    if stemmer:
        try:
            tokens = [stemmer.stem(t) for t in tokens]
        except:
            pass
    return tokens


class DocumentIndex:
    def __init__(self):
        self.documents = {}
        self.inverted_index = collections.defaultdict(dict)
        self.tf_idf = collections.defaultdict(dict)
        self.doc_vectors = {}
        self.stop_words = get_stopwords()
        self.stemmer = get_stemmer()
        self.doc_count = 0

    def add_document(self, doc_id, filepath, content):
        tokens = tokenize(content, self.stop_words, self.stemmer)
        if not tokens:
            return
        tf = collections.Counter(tokens)
        max_tf = max(tf.values()) if tf else 1
        self.documents[doc_id] = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'content': content,
            'tokens': tokens,
            'tf': tf,
            'max_tf': max_tf,
            'length': len(tokens)
        }
        for term, count in tf.items():
            self.inverted_index[term][doc_id] = count
        self.doc_count += 1

    def compute_tfidf(self):
        N = self.doc_count
        for term, doc_dict in self.inverted_index.items():
            df = len(doc_dict)
            idf = math.log((N + 1) / (df + 1)) + 1
            for doc_id, tf_count in doc_dict.items():
                max_tf = self.documents[doc_id]['max_tf']
                tf_norm = tf_count / max_tf if max_tf > 0 else 0
                self.tf_idf[term][doc_id] = tf_norm * idf

        for doc_id in self.documents:
            vec = {}
            for term in self.documents[doc_id]['tf']:
                if term in self.tf_idf and doc_id in self.tf_idf[term]:
                    vec[term] = self.tf_idf[term][doc_id]
            norm = math.sqrt(sum(v**2 for v in vec.values())) if vec else 1
            self.doc_vectors[doc_id] = {t: v/norm for t, v in vec.items()}

    def load_collection(self, folder):
        self.documents = {}
        self.inverted_index = collections.defaultdict(dict)
        self.tf_idf = collections.defaultdict(dict)
        self.doc_vectors = {}
        self.doc_count = 0

        doc_id = 1
        supported = ['.txt', '.pdf']
        for filename in sorted(os.listdir(folder)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported:
                filepath = os.path.join(folder, filename)
                content = extract_text_from_file(filepath)
                if content.strip():
                    self.add_document(doc_id, filepath, content)
                    doc_id += 1

        self.compute_tfidf()
        return self.doc_count


class SearchEngine:
    def __init__(self, index: DocumentIndex):
        self.index = index

    def preprocess_query(self, query):
        return tokenize(query, self.index.stop_words, self.index.stemmer)

    def cosine(self, query_terms):
        query_tf = collections.Counter(query_terms)
        max_qtf = max(query_tf.values()) if query_tf else 1
        N = self.index.doc_count

        query_vec = {}
        for term, count in query_tf.items():
            if term in self.index.inverted_index:
                df = len(self.index.inverted_index[term])
                idf = math.log((N + 1) / (df + 1)) + 1
                tf_norm = count / max_qtf
                query_vec[term] = tf_norm * idf

        q_norm = math.sqrt(sum(v**2 for v in query_vec.values())) if query_vec else 1
        query_vec = {t: v/q_norm for t, v in query_vec.items()}

        scores = collections.defaultdict(float)
        for term, qval in query_vec.items():
            if term in self.index.inverted_index:
                for doc_id in self.index.inverted_index[term]:
                    dval = self.index.doc_vectors.get(doc_id, {}).get(term, 0)
                    scores[doc_id] += qval * dval

        return dict(scores)

    def boolean(self, query_terms):
        if not query_terms:
            return {}
        scores = {}
        for doc_id in self.index.documents:
            doc_tokens = set(self.index.documents[doc_id]['tokens'])
            match = sum(1 for t in query_terms if t in doc_tokens)
            if match > 0:
                scores[doc_id] = match / len(query_terms)
        return scores

    def boolean_extended(self, query_terms):
        if not query_terms:
            return {}
        scores = {}
        for doc_id in self.index.documents:
            doc_tokens = set(self.index.documents[doc_id]['tokens'])
            matched = sum(1 for t in query_terms if t in doc_tokens)
            if matched > 0:
                scores[doc_id] = matched / len(query_terms)
        return scores

    def lukasiewicz(self, query_terms):
        if not query_terms:
            return {}
        scores = {}
        for doc_id in self.index.documents:
            doc_tokens = set(self.index.documents[doc_id]['tokens'])
            vals = [1.0 if t in doc_tokens else 0.0 for t in query_terms]
            luk = max(0, sum(vals) - len(vals) + 1)
            if luk > 0:
                scores[doc_id] = luk
        return scores

    def kraft(self, query_terms):
        if not query_terms:
            return {}
        scores = {}
        for doc_id in self.index.documents:
            doc_tokens = set(self.index.documents[doc_id]['tokens'])
            matched = [t for t in query_terms if t in doc_tokens]
            if matched:
                score = len(matched) / len(query_terms)
                scores[doc_id] = score ** 2
        return scores

    def jaccard(self, query_terms):
        if not query_terms:
            return {}
        query_set = set(query_terms)
        scores = {}
        for doc_id in self.index.documents:
            doc_set = set(self.index.documents[doc_id]['tokens'])
            intersection = len(query_set & doc_set)
            union = len(query_set | doc_set)
            if union > 0 and intersection > 0:
                scores[doc_id] = intersection / union
        return scores

    def dice(self, query_terms):
        if not query_terms:
            return {}
        query_set = set(query_terms)
        scores = {}
        for doc_id in self.index.documents:
            doc_set = set(self.index.documents[doc_id]['tokens'])
            intersection = len(query_set & doc_set)
            denom = len(query_set) + len(doc_set)
            if denom > 0 and intersection > 0:
                scores[doc_id] = (2 * intersection) / denom
        return scores

    def euclidean(self, query_terms):
        if not query_terms:
            return {}
        query_vec = collections.Counter(query_terms)
        scores = {}
        for doc_id in self.index.documents:
            doc_tf = self.index.documents[doc_id]['tf']
            all_terms = set(query_vec.keys()) | set(doc_tf.keys())
            dist = math.sqrt(sum((query_vec.get(t, 0) - doc_tf.get(t, 0))**2 for t in all_terms))
            max_possible = math.sqrt(sum(v**2 for v in query_vec.values()))
            if max_possible > 0:
                sim = max(0, 1 - dist / (max_possible + 1))
                if sim > 0:
                    scores[doc_id] = sim
        return scores

    def search(self, query, model='cosine'):
        query_terms = self.preprocess_query(query)
        if not query_terms:
            return []

        if model == 'cosine':
            scores = self.cosine(query_terms)
        elif model == 'boolean':
            scores = self.boolean(query_terms)
        elif model == 'boolean_extended':
            scores = self.boolean_extended(query_terms)
        elif model == 'lukasiewicz':
            scores = self.lukasiewicz(query_terms)
        elif model == 'kraft':
            scores = self.kraft(query_terms)
        elif model == 'jaccard':
            scores = self.jaccard(query_terms)
        elif model == 'dice':
            scores = self.dice(query_terms)
        elif model == 'euclidean':
            scores = self.euclidean(query_terms)
        else:
            scores = self.cosine(query_terms)

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results


app = Flask(__name__, template_folder='templates', static_folder='static')
index = DocumentIndex()
engine = SearchEngine(index)
collection_folder = os.path.join(os.path.dirname(__file__), "documents")


def create_sample_docs():
    samples = [
        ("introduction_ia.txt",
         """Introduction à l'Intelligence Artificielle
L'intelligence artificielle est un domaine de l'informatique qui vise à créer des machines capables de simuler l'intelligence humaine. Le machine learning et le deep learning sont des sous-domaines de l'IA. Les réseaux de neurones artificiels s'inspirent du cerveau humain. L'IA est utilisée dans de nombreux domaines : la santé, les transports, la finance, l'éducation et les loisirs. Les algorithmes d'apprentissage automatique permettent aux ordinateurs d'apprendre à partir de données sans être explicitement programmés."""),
        ("python_programming.txt",
         """Guide complet Python pour la Data Science
Python est un langage de programmation puissant pour la science des données et l'apprentissage automatique. Les bibliothèques comme numpy, pandas et scikit-learn sont essentielles. Python est simple, lisible et portable. Il supporte la programmation orientée objet et fonctionnelle. Les data scientists utilisent Python pour l'analyse de données, la visualisation, et le développement de modèles de machine learning. TensorFlow et PyTorch sont des frameworks populaires pour le deep learning en Python."""),
        ("machine_learning.txt",
         """Machine Learning et Algorithmes
Le machine learning est une branche de l'intelligence artificielle qui permet aux systèmes d'apprendre automatiquement. Les algorithmes supervisés utilisent des données étiquetées pour l'entraînement. Les algorithmes non supervisés découvrent des structures cachées. La classification, la régression et le clustering sont des tâches fondamentales. Les forêts aléatoires, les SVM et les réseaux de neurones sont des méthodes populaires. L'évaluation des modèles utilise des métriques comme la précision, le rappel et le F1-score."""),
        ("bases_de_donnees.txt",
         """Systèmes de Gestion de Bases de Données
Les bases de données relationnelles utilisent SQL pour stocker et interroger des données structurées. MySQL, PostgreSQL et Oracle sont des SGBD populaires. Les bases NoSQL comme MongoDB et Redis offrent flexibilité et scalabilité. Les transactions ACID garantissent la cohérence des données. L'indexation améliore les performances des requêtes. La normalisation réduit la redondance des données dans les bases relationnelles."""),
        ("reseaux_informatiques.txt",
         """Réseaux Informatiques et Protocoles
Les réseaux informatiques permettent la communication entre ordinateurs. Le protocole TCP/IP est la base d'Internet. DNS traduit les noms de domaine en adresses IP. HTTP et HTTPS permettent la navigation web. Les réseaux locaux LAN utilisent Ethernet et WiFi. Les firewalls protègent les réseaux contre les intrusions. La sécurité réseau inclut le chiffrement, l'authentification et le contrôle d'accès."""),
        ("algorithmes_tri.txt",
         """Algorithmes de Tri et Complexité
Les algorithmes de tri organisent les données selon un ordre défini. Le tri à bulles a une complexité O(n²). Le tri rapide (quicksort) a une complexité moyenne O(n log n). Le tri fusion (mergesort) est stable et garantit O(n log n). Les structures de données comme les arbres binaires de recherche permettent des insertions et recherches efficaces. La complexité algorithmique mesure l'efficacité en temps et en espace mémoire."""),
        ("securite_informatique.txt",
         """Sécurité Informatique et Cryptographie
La sécurité informatique protège les systèmes contre les attaques malveillantes. La cryptographie symétrique utilise une seule clé pour chiffrer et déchiffrer. La cryptographie asymétrique utilise une paire de clés publique/privée. SSL/TLS sécurise les communications sur Internet. Les attaques par injection SQL et XSS menacent les applications web. L'authentification multi-facteurs renforce la sécurité des accès."""),
        ("developpement_web.txt",
         """Développement Web Moderne
Le développement web utilise HTML, CSS et JavaScript pour créer des sites interactifs. Les frameworks React, Vue et Angular facilitent le développement frontend. Node.js permet d'utiliser JavaScript côté serveur. Les API REST permettent la communication entre frontend et backend. Git est essentiel pour la gestion de versions du code source. Docker et Kubernetes facilitent le déploiement des applications web."""),
    ]
    for filename, content in samples:
        filepath = os.path.join(collection_folder, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)


@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '').strip()
    model = data.get('model', 'boolean_extended')
    
    if not query:
        return jsonify({'results': [], 'count': 0, 'elapsed': 0})
    
    start = time.time()
    results = engine.search(query, model)
    elapsed = time.time() - start
    
    formatted_results = []
    query_words = query.lower().split()
    
    for rank, (doc_id, score) in enumerate(results[:20]):
        doc = index.documents.get(doc_id)
        if not doc:
            continue
        
        score_pct = min(100, round(score * 100, 1))
        snippet = highlight_text(doc['content'], query_words)
        
        formatted_results.append({
            'rank': rank + 1,
            'filename': doc['filename'],
            'score': score_pct,
            'snippet': snippet,
            'content': doc['content'],
            'filepath': doc['filepath']
        })
    
    return jsonify({
        'results': formatted_results,
        'count': len(formatted_results),
        'elapsed': round(elapsed, 2),
        'model': model
    })


def highlight_text(text, query_words, max_len=180):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    lower_text = text.lower()
    best_pos = 0
    for word in query_words:
        pos = lower_text.find(word.lower())
        if pos != -1:
            best_pos = max(0, pos - 40)
            break
    
    snippet = text[best_pos:best_pos + max_len]
    if best_pos > 0:
        snippet = "..." + snippet
    if best_pos + max_len < len(text):
        snippet = snippet + "..."
    
    for word in query_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        snippet = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", snippet)
    
    return snippet


@app.route('/api/document/<int:doc_id>')
def get_document(doc_id):
    doc = index.documents.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    return jsonify({
        'filename': doc['filename'],
        'filepath': doc['filepath'],
        'content': doc['content']
    })


if __name__ == '__main__':
    if not os.path.exists(collection_folder):
        os.makedirs(collection_folder)
    
    create_sample_docs()
    index.load_collection(collection_folder)
    app.run(debug=True, port=5000)
