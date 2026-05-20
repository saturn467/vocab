import pandas as pd


class WordLoader:

    def load_excel(self, file_path):

        df = pd.read_excel(file_path, engine="openpyxl")

        word_column = "lemme"
        rank_column = "freq"

        words = []
        ranks = {}

        for _, row in df.iterrows():

            word = str(row[word_column]).strip()
            rank = int(row[rank_column])

            words.append(word)
            ranks[word] = rank

        return words, ranks