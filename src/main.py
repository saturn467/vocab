import uuid

from loader import WordLoader
from embeddings import EmbeddingGenerator
from clustering import Clusterer
from difficulty import DifficultyClassifier
from category_classifier import CategoryClassifier
from exporter import Exporter


def main():

    print("Loading Excel vocabulary...")

    loader = WordLoader()

    words, ranks = loader.load_excel("data/words.xlsx")

    print(f"{len(words)} words loaded")


    # Combine all terms (currently just words)
    all_terms = words


    # Generate embeddings
    print("Generating embeddings...")

    embedder = EmbeddingGenerator()

    embeddings = embedder.encode(all_terms)


    # Cluster words into topics
    print("Clustering words...")

    clusterer = Clusterer()

    clusters = clusterer.cluster(embeddings)


    difficulty_classifier = DifficultyClassifier()

    category_classifier = CategoryClassifier()


    topics = []
    word_rows = []


    for cluster_id, indices in clusters.items():

        cluster_terms = [all_terms[i] for i in indices]

        cluster_ranks = [ranks[t] for t in cluster_terms]

        level = difficulty_classifier.classify(cluster_ranks)

        topic_name = f"topic_{cluster_id}_{level}"

        topic_id = str(uuid.uuid4())


        topics.append({
            "id": topic_id,
            "name": topic_name,
            "difficulty": level
        })


        for term in cluster_terms:

            theme, category = category_classifier.classify(term)

            word_rows.append({
                "id": str(uuid.uuid4()),
                "term": term,
                "frequency_rank": ranks[term],
                "theme": theme,
                "category": category,
                "topic_id": topic_id
            })


    exporter = Exporter()

    exporter.export(topics, word_rows)

    print("Vocabulary catalog created!")


if __name__ == "__main__":
    main()