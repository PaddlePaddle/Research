# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from paddlemetrics.functional.audio.pesq import pesq
from paddlemetrics.functional.audio.pit import pit, pit_permutate
from paddlemetrics.functional.audio.si_sdr import si_sdr
from paddlemetrics.functional.audio.si_snr import si_snr
from paddlemetrics.functional.audio.snr import snr
from paddlemetrics.functional.audio.stoi import stoi
from paddlemetrics.functional.classification.accuracy import accuracy
from paddlemetrics.functional.classification.auc import auc
from paddlemetrics.functional.classification.auroc import auroc
from paddlemetrics.functional.classification.average_precision import average_precision
from paddlemetrics.functional.classification.calibration_error import calibration_error
from paddlemetrics.functional.classification.cohen_kappa import cohen_kappa
from paddlemetrics.functional.classification.confusion_matrix import confusion_matrix
from paddlemetrics.functional.classification.dice import dice_score
from paddlemetrics.functional.classification.f_beta import f1, fbeta
from paddlemetrics.functional.classification.hamming_distance import hamming_distance
from paddlemetrics.functional.classification.hinge import hinge
from paddlemetrics.functional.classification.iou import iou
from paddlemetrics.functional.classification.kl_divergence import kl_divergence
from paddlemetrics.functional.classification.matthews_corrcoef import matthews_corrcoef
from paddlemetrics.functional.classification.precision_recall import precision, precision_recall, recall
from paddlemetrics.functional.classification.precision_recall_curve import precision_recall_curve
from paddlemetrics.functional.classification.roc import roc
from paddlemetrics.functional.classification.specificity import specificity
from paddlemetrics.functional.classification.stat_scores import stat_scores
from paddlemetrics.functional.image.gradients import image_gradients
from paddlemetrics.functional.image.psnr import psnr
from paddlemetrics.functional.image.ssim import ssim
from paddlemetrics.functional.pairwise.cosine import pairwise_cosine_similarity
from paddlemetrics.functional.pairwise.euclidean import pairwise_euclidean_distance
from paddlemetrics.functional.pairwise.linear import pairwise_linear_similarity
from paddlemetrics.functional.pairwise.manhatten import pairwise_manhatten_distance
from paddlemetrics.functional.regression.cosine_similarity import cosine_similarity
from paddlemetrics.functional.regression.explained_variance import explained_variance
from paddlemetrics.functional.regression.mean_absolute_error import mean_absolute_error
from paddlemetrics.functional.regression.mean_absolute_percentage_error import mean_absolute_percentage_error
from paddlemetrics.functional.regression.mean_squared_error import mean_squared_error
from paddlemetrics.functional.regression.mean_squared_log_error import mean_squared_log_error
from paddlemetrics.functional.regression.pearson import pearson_corrcoef
from paddlemetrics.functional.regression.r2 import r2_score
from paddlemetrics.functional.regression.spearman import spearman_corrcoef
from paddlemetrics.functional.regression.symmetric_mean_absolute_percentage_error import (
    symmetric_mean_absolute_percentage_error,
)
from paddlemetrics.functional.regression.tweedie_deviance import tweedie_deviance_score
from paddlemetrics.functional.retrieval.average_precision import retrieval_average_precision
from paddlemetrics.functional.retrieval.fall_out import retrieval_fall_out
from paddlemetrics.functional.retrieval.hit_rate import retrieval_hit_rate
from paddlemetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from paddlemetrics.functional.retrieval.precision import retrieval_precision
from paddlemetrics.functional.retrieval.r_precision import retrieval_r_precision
from paddlemetrics.functional.retrieval.recall import retrieval_recall
from paddlemetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank
from paddlemetrics.functional.self_supervised import embedding_similarity
#from paddlemetrics.functional.text.bert import bert_score
from paddlemetrics.functional.text.bleu import bleu_score
from paddlemetrics.functional.text.rouge import rouge_score
from paddlemetrics.functional.text.sacre_bleu import sacre_bleu_score
from paddlemetrics.functional.text.wer import wer

__all__ = [
    "accuracy",
    "auc",
    "auroc",
    "average_precision",
#    "bert_score",
    "bleu_score",
    "calibration_error",
    "cohen_kappa",
    "confusion_matrix",
    "cosine_similarity",
    "tweedie_deviance_score",
    "dice_score",
    "embedding_similarity",
    "explained_variance",
    "f1",
    "fbeta",
    "hamming_distance",
    "hinge",
    "image_gradients",
    "iou",
    "kl_divergence",
    "kldivergence",
    "matthews_corrcoef",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "pairwise_cosine_similarity",
    "pairwise_euclidean_distance",
    "pairwise_linear_similarity",
    "pairwise_manhatten_distance",
    "pearson_corrcoef",
    "pesq",
    "pit",
    "pit_permutate",
    "precision",
    "precision_recall",
    "precision_recall_curve",
    "psnr",
    "r2_score",
    "r2score",
    "recall",
    "retrieval_average_precision",
    "retrieval_fall_out",
    "retrieval_hit_rate",
    "retrieval_normalized_dcg",
    "retrieval_precision",
    "retrieval_r_precision",
    "retrieval_recall",
    "retrieval_reciprocal_rank",
    "roc",
    "rouge_score",
    "sacre_bleu_score",
    "si_sdr",
    "si_snr",
    "snr",
    "spearman_corrcoef",
    "specificity",
    "ssim",
    "stat_scores",
    "stoi",
    "symmetric_mean_absolute_percentage_error",
    "wer",
]
