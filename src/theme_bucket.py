from collections import defaultdict

class ThemeBucketBuilder:

    def build(self, words, theme_classifier):

        buckets = defaultdict(list)

        for word in words:

            themes = theme_classifier.classify_word(word)

            for theme in themes:
                buckets[theme].append(word)

        return buckets