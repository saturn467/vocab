class CorpusLoader:

    def load(self, path="data/corpus.txt"):

        with open(path) as f:
            return [line.strip() for line in f if line.strip()]