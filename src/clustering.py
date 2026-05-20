from sklearn.cluster import KMeans
import math

class Clusterer:
    def __init__(self, words_per_topic: int):
        self.words_per_topic = words_per_topic

    def cluster(self, words: list[str], embeddings):
        num_clusters = math.ceil(len(words) / self.words_per_topic)

        model = KMeans(n_clusters=num_clusters, random_state=42)
        labels = model.fit_predict(embeddings)

        clusters = {}
        for word, label in zip(words, labels):
            clusters.setdefault(label, []).append(word)

        return list(clusters.values())