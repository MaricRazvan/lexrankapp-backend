import numpy as np
import json
from preprocessing import TextPreprocessor

class LexRankSummarizer:
    def __init__(self, embedder, stopwords, threshold=0.1, damping=0.15, max_iter=100):
        self.embedder = embedder
        self.threshold = threshold
        self.damping = damping
        self.max_iter = max_iter
        self.stopwords = stopwords
        self.preprocessor = TextPreprocessor()

    def cosine_similarity(self, vec1, vec2):
        dot = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return 0 if norm1 == 0 or norm2 == 0 else dot / (norm1 * norm2)

    def build_similarity_matrix(self, matrix):
        n = matrix.shape[0]
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim = self.cosine_similarity(matrix[i], matrix[j])
                    if sim > self.threshold:
                        sim_matrix[i][j] = sim
        return sim_matrix

    def apply_lexrank(self, sim_matrix, epsilon=1e-4):
        n = sim_matrix.shape[0]
        scores = np.ones(n) / n
        row_sums = sim_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        trans_matrix = sim_matrix / row_sums[:, np.newaxis]

        for _ in range(self.max_iter):
            prev_scores = scores.copy()
            scores = self.damping / n + (1 - self.damping) * np.dot(trans_matrix.T, prev_scores)
            if np.linalg.norm(scores - prev_scores) < epsilon:
                break
        return scores

    def summarize(self, text, compression_rate=0.3):
        original, processed = self.preprocessor.preprocess(text, self.stopwords)
        if len(original) <= 1:
            return text, original, [1.0]

        matrix = self.embedder.embed(processed)
        sim_matrix = self.build_similarity_matrix(matrix)
        scores = self.apply_lexrank(sim_matrix)
        top_n = max(1, int(round(compression_rate * len(original))))
        top_idxs = sorted([int(i) for i in np.argsort(-scores)[:top_n]])
        summary = " ".join([original[i] for i in top_idxs])
        return summary, original, scores

    def summarize_to_json(self, text, compression_rate=0.3):
        summary, original, scores = self.summarize(text, compression_rate)
        top_n = max(1, int(round(compression_rate * len(original))))
        top_idxs = sorted([int(i) for i in np.argsort(-scores)[:top_n]])

        result = {
            "summary": summary,
            "sentences": original,
            "scores": [float(f"{s:.4f}") for s in scores],
            "selected_indices": top_idxs,
            "selected_sentences": [original[i] for i in top_idxs]
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
