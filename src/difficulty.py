class DifficultyClassifier:

    def classify(self, ranks):

        avg_rank = sum(ranks) / len(ranks)

        if avg_rank <= 500:
            return "basic"

        if avg_rank <= 1200:
            return "intermediate"

        return "advanced"