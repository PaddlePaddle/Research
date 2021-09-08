import paddle
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelBinarizer


def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


class Metrics(object):

    def __init__(self):
        super().__init__()
        self.PAD = 0

    def hits_score(self, ground_truth, predictions, k=5):
        score = 0
        for y_true, y_score in zip(ground_truth, predictions):
            topk = y_score.argsort()[-k:][::-1]
            if y_true in topk:
                score += 1
        return score / len(ground_truth)

    def ndcg_score(self, ground_truth, predictions, k=5):
        """Normalized discounted cumulative gain (NDCG) at rank K.

        Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
        recommendation system based on the graded relevance of the recommended
        entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
        ranking of the entities.

        Parameters
        ----------
        ground_truth : array, shape = [n_samples]
            Ground truth (true labels represended as integers).
        predictions : array, shape = [n_samples, n_classes]
            Predicted probabilities.
        k : int
            Rank.

        Returns
        -------
        score : float

        Example
        -------
        >>> ground_truth = [1, 0, 2]
        >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
        >>> score = ndcg_score(ground_truth, predictions, k=2)
        1.0
        >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
        >>> score = ndcg_score(ground_truth, predictions, k=2)
        0.6666666666
        """
        T = np.zeros(shape=predictions.shape)
        idx = np.arange(T.shape[0])
        T[idx, ground_truth] = 1
        scores = []
        for y_true, y_score in zip(T, predictions):
            actual = dcg_score(y_true, y_score, k)
            best = dcg_score(y_true, y_true, k)
            score = float(actual) / float(best)
            scores.append(score)
        return np.mean(scores)

    def compute_metric(self, y_prob, y_true, session_len=101):
        session = []
        y_prob_session = []
        y_true_session = []
        for i, (y_, y) in enumerate(zip(y_prob, y_true)):
            session.append(y_)
            if y == 1:
                y_true_session.append(i % session_len)
            if (i + 1) % session_len == 0:
                y_prob_session.append(session)
                session = []
        y_true_session = np.array(y_true_session)
        y_prob_session = np.array(y_prob_session)
        hits5 = self.hits_score(y_true_session, y_prob_session, k=5)
        hits10 = self.hits_score(y_true_session, y_prob_session, k=10)
        ndcg5 = self.ndcg_score(y_true_session, y_prob_session, k=5)
        ndcg10 = self.ndcg_score(y_true_session, y_prob_session, k=10)
        return hits5, hits10, ndcg5, ndcg10
