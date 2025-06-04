from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
import math
import torch
from transformers import AutoTokenizer, AutoModel

class AbstractEmbedder(ABC):
    @abstractmethod
    def embed(self, processed_sentences):
        pass

class TFIDFEmbedder(AbstractEmbedder):
    def calculate_tf(self, sentence):
        tf_dict = {}
        token_count = len(sentence)
        term_counts = Counter(sentence)
        for term, count in term_counts.items():
            tf_dict[term] = count / token_count
        return tf_dict

    def calculate_idf(self, processed_sentences):
        idf_dict = {}
        num_sentences = len(processed_sentences)
        all_terms = set(term for s in processed_sentences for term in s)

        for term in all_terms:
            doc_count = sum(1 for sentence in processed_sentences if term in sentence)
            idf_dict[term] = math.log(num_sentences / (1 + doc_count))

        return idf_dict

    def embed(self, processed_sentences):
        idf_dict = self.calculate_idf(processed_sentences)
        all_terms = sorted(idf_dict.keys())
        tfidf_matrix = np.zeros((len(processed_sentences), len(all_terms)))

        for i, sentence in enumerate(processed_sentences):
            tf_dict = self.calculate_tf(sentence)
            for j, term in enumerate(all_terms):
                if term in tf_dict:
                    tfidf_matrix[i][j] = tf_dict[term] * idf_dict[term]

        return tfidf_matrix

class RoBERTaEmbedder(AbstractEmbedder):
    def __init__(self, model_name="dumitrescustefan/bert-base-romanian-cased-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, processed_sentences):
        with torch.no_grad():
            embeddings = []
            for tokens in processed_sentences:
                text = " ".join(tokens)
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
            return np.vstack(embeddings)
