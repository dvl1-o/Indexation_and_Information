import tkinter as tk
from tkinter import ttk, messagebox
import os
import math
import re
import time
import collections

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, SnowballStemmer
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
            try:
                return PorterStemmer()
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


class SRIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SRI Recherche")
        self.root.geometry("960x720")
        self.root.configure(bg="#f5f5f5")

        self.index = DocumentIndex()
        self.engine = SearchEngine(self.index)
        self.collection_folder = os.path.join(os.path.dirname(__file__), "documents")
        self.search_results = []
        self.model_var = tk.StringVar(value="boolean_extended")

        self._build_ui()
        self._load_collection()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#ffffff", pady=18)
        header.pack(fill=tk.X)

        title_frame = tk.Frame(header, bg="#ffffff")
        title_frame.pack()

        tk.Label(title_frame, text="SRI", font=("Arial", 38, "bold"),
                 fg="#1a73e8", bg="#ffffff").pack(side=tk.LEFT)
        tk.Label(title_frame, text=" Recherche", font=("Arial", 32),
                 fg="#e53935", bg="#ffffff").pack(side=tk.LEFT)

        search_frame = tk.Frame(self.root, bg="#f5f5f5", pady=12)
        search_frame.pack(fill=tk.X, padx=60)

        self.search_entry = tk.Entry(search_frame, font=("Arial", 14),
                                     relief=tk.SOLID, bd=1, width=45)
        self.search_entry.pack(side=tk.LEFT, ipady=7, padx=(0, 6))
        self.search_entry.bind("<Return>", lambda e: self._do_search())

        search_btn = tk.Button(search_frame, text="🔍", font=("Arial", 13),
                                command=self._do_search, bg="#1a73e8", fg="white",
                                relief=tk.FLAT, cursor="hand2", padx=10, pady=4)
        search_btn.pack(side=tk.LEFT)

        model_outer = tk.LabelFrame(self.root, text="  MODÈLE DE RECHERCHE  ",
                                    font=("Arial", 9, "bold"), bg="#f5f5f5",
                                    fg="#555555", padx=10, pady=8)
        model_outer.pack(fill=tk.X, padx=60, pady=(0, 8))

        models = [
            ("Cosinus", "cosine"),
            ("Booléen", "boolean"),
            ("Booléen étendu", "boolean_extended"),
            ("Lukasiewicz", "lukasiewicz"),
            ("Kraft", "kraft"),
            ("Jaccard", "jaccard"),
            ("Dice", "dice"),
            ("Euclidienne", "euclidean"),
        ]

        row1 = tk.Frame(model_outer, bg="#f5f5f5")
        row1.pack()
        row2 = tk.Frame(model_outer, bg="#f5f5f5")
        row2.pack()

        icons = {
            "cosine": "📐", "boolean": "📋", "boolean_extended": "🌐",
            "lukasiewicz": "🔢", "kraft": "📄", "jaccard": "📊",
            "dice": "🎲", "euclidean": "📏"
        }

        for i, (label, val) in enumerate(models):
            frame = row1 if i < 5 else row2
            ico = icons.get(val, "")
            rb = tk.Radiobutton(frame, text=f"{ico} {label}", variable=self.model_var,
                                value=val, font=("Arial", 10), bg="#f5f5f5",
                                activebackground="#f5f5f5", cursor="hand2",
                                command=self._do_search)
            rb.pack(side=tk.LEFT, padx=10, pady=2)

        self.status_label = tk.Label(self.root, text="", font=("Arial", 9),
                                     fg="#666666", bg="#f5f5f5", anchor='w')
        self.status_label.pack(fill=tk.X, padx=60, pady=(4, 0))

        results_outer = tk.Frame(self.root, bg="#f5f5f5")
        results_outer.pack(fill=tk.BOTH, expand=True, padx=60, pady=4)

        canvas = tk.Canvas(results_outer, bg="#f5f5f5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.results_frame = tk.Frame(canvas, bg="#f5f5f5")
        self.canvas_window = canvas.create_window((0, 0), window=self.results_frame, anchor="nw")

        self.results_frame.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(
            self.canvas_window, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        status_bar = tk.Frame(self.root, bg="#e0e0e0", pady=3)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.collection_status = tk.Label(status_bar, text="Chargement...",
                                          font=("Arial", 9), fg="#555555", bg="#e0e0e0")
        self.collection_status.pack(side=tk.LEFT, padx=10)

    def _load_collection(self):
        if not os.path.exists(self.collection_folder):
            os.makedirs(self.collection_folder)
            self._create_sample_docs()

        count = self.index.load_collection(self.collection_folder)
        self.collection_status.config(
            text=f"Collection chargée : {count} document(s) dans '{self.collection_folder}'"
        )

    def _create_sample_docs(self):
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
            with open(os.path.join(self.collection_folder, filename), 'w', encoding='utf-8') as f:
                f.write(content)

    def _do_search(self):
        query = self.search_entry.get().strip()
        if not query:
            return

        model = self.model_var.get()
        start = time.time()
        results = self.engine.search(query, model)
        elapsed = time.time() - start

        model_names = {
            'cosine': 'Cosinus', 'boolean': 'Booléen',
            'boolean_extended': 'Booléen étendu', 'lukasiewicz': 'Lukasiewicz',
            'kraft': 'Kraft', 'jaccard': 'Jaccard',
            'dice': 'Dice', 'euclidean': 'Euclidienne'
        }

        self.search_results = results
        self.status_label.config(
            text=f"{len(results)} résultat(s) ({elapsed:.2f} secondes) - Modèle : {model_names.get(model, model)}"
        )
        self._display_results(results, query)

    def _highlight_text(self, text, query_words, max_len=180):
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
        return snippet

    def _display_results(self, results, query):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if not results:
            tk.Label(self.results_frame, text="Aucun résultat trouvé.",
                     font=("Arial", 12), fg="#888888", bg="#f5f5f5").pack(pady=30)
            return

        query_words = query.lower().split()

        for rank, (doc_id, score) in enumerate(results[:20]):
            doc = self.index.documents.get(doc_id)
            if not doc:
                continue

            score_pct = min(100, round(score * 100, 1))

            card = tk.Frame(self.results_frame, bg="#ffffff", relief=tk.FLAT,
                            bd=0, pady=10, padx=14, cursor="hand2")
            card.pack(fill=tk.X, pady=4)
            card.bind("<Button-1>", lambda e, d=doc: self._show_document(d))

            top_row = tk.Frame(card, bg="#ffffff")
            top_row.pack(fill=tk.X)

            ext = os.path.splitext(doc['filename'])[1].upper().lstrip('.')
            ext_color = "#d32f2f" if ext == "PDF" else "#1565c0"
            tk.Label(top_row, text=f"📄 {doc['filename']} • {ext} • Score: {score_pct}%",
                     font=("Arial", 8), fg="#888888", bg="#ffffff").pack(side=tk.LEFT)

            title_text = os.path.splitext(doc['filename'])[0].replace('_', ' ').title()
            first_line = doc['content'].strip().split('\n')[0][:80]
            if len(first_line) > 10:
                title_text = first_line

            title_lbl = tk.Label(card, text=title_text, font=("Arial", 13, "bold"),
                                 fg="#1a0dab", bg="#ffffff", cursor="hand2", anchor='w')
            title_lbl.pack(fill=tk.X)
            title_lbl.bind("<Button-1>", lambda e, d=doc: self._show_document(d))

            snippet = self._highlight_text(doc['content'], query_words)
            for word in query_words:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                snippet = pattern.sub(lambda m: f"[{m.group()}]", snippet)

            snippet_lbl = tk.Label(card, text=snippet, font=("Arial", 10),
                                   fg="#444444", bg="#ffffff", wraplength=800,
                                   justify=tk.LEFT, anchor='w')
            snippet_lbl.pack(fill=tk.X)

            score_bar_frame = tk.Frame(card, bg="#ffffff")
            score_bar_frame.pack(fill=tk.X, pady=(4, 0))

            score_color = "#34a853" if score_pct >= 50 else "#fbbc04" if score_pct >= 20 else "#ea4335"
            score_tag = tk.Label(score_bar_frame,
                                 text=f"  Pertinence: {score_pct}%  ",
                                 font=("Arial", 8, "bold"),
                                 fg="white", bg=score_color)
            score_tag.pack(side=tk.LEFT)

            separator = tk.Frame(self.results_frame, height=1, bg="#e0e0e0")
            separator.pack(fill=tk.X, pady=0)

    def _show_document(self, doc):
        popup = tk.Toplevel(self.root)
        popup.title(doc['filename'])
        popup.geometry("680x500")
        popup.configure(bg="#ffffff")
        popup.grab_set()

        header = tk.Frame(popup, bg="#5c6bc0", padx=16, pady=12)
        header.pack(fill=tk.X)

        tk.Label(header, text=os.path.splitext(doc['filename'])[0].replace('_', ' ').title(),
                 font=("Arial", 14, "bold"), fg="white", bg="#5c6bc0").pack(side=tk.LEFT)

        close_btn = tk.Button(header, text="✕", font=("Arial", 11, "bold"),
                              command=popup.destroy, bg="#5c6bc0", fg="white",
                              relief=tk.FLAT, cursor="hand2")
        close_btn.pack(side=tk.RIGHT)

        meta_frame = tk.Frame(popup, bg="#f5f5f5", padx=16, pady=8)
        meta_frame.pack(fill=tk.X)

        ext = os.path.splitext(doc['filename'])[1].upper().lstrip('.')
        tk.Label(meta_frame, text=f"📄 {ext}",
                 font=("Arial", 9), fg="#888888", bg="#f5f5f5").pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(meta_frame, text=f"📁 {doc['filename']}",
                 font=("Arial", 9), fg="#888888", bg="#f5f5f5").pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(meta_frame, text=f"🔢 ID: {list(self.index.documents.keys())[list(self.index.documents.values()).index(doc)]}",
                 font=("Arial", 9), fg="#888888", bg="#f5f5f5").pack(side=tk.LEFT)

        text_frame = tk.Frame(popup, bg="#ffffff", padx=16, pady=12)
        text_frame.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(text_frame, font=("Arial", 10), wrap=tk.WORD,
                              relief=tk.FLAT, bg="#ffffff", fg="#333333",
                              padx=8, pady=8)
        scroll = ttk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(fill=tk.BOTH, expand=True)

        text_widget.insert(tk.END, doc['content'])
        text_widget.configure(state=tk.DISABLED)

        footer = tk.Label(popup,
                          text=f"Fichier: {doc['filepath']}  •  Cliquez en dehors ou sur la croix pour fermer",
                          font=("Arial", 8), fg="#888888", bg="#f0f0f0", pady=4)
        footer.pack(fill=tk.X)

        popup.bind("<Escape>", lambda e: popup.destroy())
        popup.bind("<Button-1>", lambda e: popup.destroy() if e.widget == popup else None)


def main():
    root = tk.Tk()
    app = SRIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
