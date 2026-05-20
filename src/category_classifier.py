from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import CATEGORIES


class CategoryClassifier:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.category_map = {}
        self.category_embeddings = []

        for theme, categories in CATEGORIES.items():
            for c in categories:
                self.category_map[c] = theme
                self.category_embeddings.append(c)

        self.category_vectors = self.model.encode(self.category_embeddings)

    def classify(self, word):

        embedding = self.model.encode([word])[0]

        scores = cosine_similarity(
            [embedding],
            self.category_vectors
        )[0]

        best_index = scores.argmax()

        category = self.category_embeddings[best_index]

        theme = self.category_map[category]

        return theme, category