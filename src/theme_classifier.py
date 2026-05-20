from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import THEMES

class ThemeClassifier:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.theme_embeddings = self.model.encode(THEMES)

    def classify_word(self, word):

        word_embedding = self.model.encode([word])[0]

        scores = cosine_similarity(
            [word_embedding],
            self.theme_embeddings
        )[0]

        themes = []

        for i, s in enumerate(scores):
            if s > 0.35:
                themes.append(THEMES[i])

        if not themes:
            themes.append("Culture")

        return themes