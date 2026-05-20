from collections import Counter
import math


class PhraseMiner:

    def __init__(self, min_count=10):
        self.min_count = min_count

    def mine_phrases(self, corpus):

        unigram_counts = Counter()
        bigram_counts = Counter()

        for sentence in corpus:

            words = sentence.lower().split()

            for w in words:
                unigram_counts[w] += 1

            for i in range(len(words) - 1):
                bigram = (words[i], words[i+1])
                bigram_counts[bigram] += 1

        phrases = []

        total_words = sum(unigram_counts.values())

        for (w1, w2), count in bigram_counts.items():

            if count < self.min_count:
                continue

            p_w1 = unigram_counts[w1] / total_words
            p_w2 = unigram_counts[w2] / total_words
            p_w1w2 = count / total_words

            pmi = math.log2(p_w1w2 / (p_w1 * p_w2))

            if pmi > 3:
                phrases.append(f"{w1} {w2}")

        return phrases