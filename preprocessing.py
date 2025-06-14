# Required imports
import re
import stanza

class TextPreprocessor:
    def __init__(self):
        stanza.download('ro', verbose=False)  # Download Romanian model (only once)
        self.nlp = stanza.Pipeline('ro', processors='tokenize,pos,lemma', use_gpu=False)

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

            doc = self.nlp(sentence)
            lemmas = [
                word.lemma.lower()
                for sent in doc.sentences
                for word in sent.words
                if word.lemma and word.lemma.lower() not in stopwords and len(word.lemma) > 1
            ]

            processed_sentences.append(lemmas)

        return original_sentences, processed_sentences
