# Required imports
import re
import snowballstemmer

class TextPreprocessor:
    def __init__(self):
        self.stemmer = snowballstemmer.stemmer('romanian')

    def normalize_diacritics(self, text):
        replacements = {'ş': 'ș', 'Ş': 'Ș', 'ţ': 'ț', 'Ţ': 'Ț'}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def preprocess(self, text, stopwords):
        text = self.normalize_diacritics(text)

        # First: handle sentence splits with punctuation and capital letters (includes optional quote)
        text = re.sub(r'([.!?])(["”»]?)(\s+)(?=[A-ZĂÂÎȘȚ])', r'\1\2\n', text)

        # Second: force split when a quote starts a new sentence after a period, even if punctuation is missing
        text = re.sub(r'(\.)\s*(")(?=[A-ZĂÂÎȘȚ])', r'\1\n\2', text)

        raw_sentences = text.strip().split('\n')

        original_sentences, processed_sentences = [], []
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if len(sentence) < 2:
                continue
            original_sentences.append(sentence)

            tokens = re.findall(r'\b\w+\b', sentence.lower())
            tokens = [t for t in tokens if t not in stopwords]
            tokens = self.stemmer.stemWords(tokens)
            tokens = [t for t in tokens if len(t) > 1]

            processed_sentences.append(tokens)

        return original_sentences, processed_sentences
