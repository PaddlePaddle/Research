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

import pytest
import paddleext.torchapi as B

from tests.text.helpers import TextTester
from paddlemetrics.functional.text.sacre_bleu import sacre_bleu_score
from paddlemetrics.text.sacre_bleu import SacreBLEUScore
from paddlemetrics.utilities.imports import _SACREBLEU_AVAILABLE

if _SACREBLEU_AVAILABLE:
    from sacrebleu.metrics import BLEU

# example taken from https://github.com/mjpost/sacrebleu
REFERENCES = (
    # First set of references
    ("The dog bit the man.", "It was not unexpected.", "The man bit him first."),
    # Second set of references
    ("The dog had bit the man.", "No one was surprised.", "The man had bitten the dog."),
)

HYPOTHESES = ("The dog bit the man.", "It wasn't surprising.", "The man had just bitten him.")

TOKENIZERS = ("none", "13a", "zh", "intl", "char")

ROUND_N_DIGITS = 4


def metrics_score_fn(targets, preds, tokenize):
    metrics_score = sacre_bleu_score(targets, preds, tokenize=tokenize)
    # rescale to 0-100 and round to 4 decimals to match blue
    metrics_score_normed = B.round(100 * metrics_score * 10 ** ROUND_N_DIGITS) / 10 ** ROUND_N_DIGITS
    return metrics_score_normed


@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (HYPOTHESES, REFERENCES),
    ],
)
@pytest.mark.parametrize("tokenize", TOKENIZERS)
@pytest.mark.skipif(not _SACREBLEU_AVAILABLE, reason="test requires sacrebleu")
class TestSacreBLEUScore(TextTester):
    def test_sacrebleu_score_functional(self, preds, targets, tokenize):
        sacrebleu_metrics = BLEU(tokenize=tokenize)
        original_score = B.tensor(round(sacrebleu_metrics.corpus_score(preds, targets).score, ROUND_N_DIGITS))

        metrics_targets = [[ref[i] for ref in targets] for i in range(len(targets[0]))]
        metrics_score = metrics_score_fn(metrics_targets, preds, tokenize)
        assert metrics_score == original_score

    def test_sacrebleu_score_metrics(self, preds, targets, tokenize):
        sacrebleu_metrics = BLEU(tokenize=tokenize)
        original_score = B.tensor(round(sacrebleu_metrics.corpus_score(preds, targets).score, ROUND_N_DIGITS))

        metrics_targets = [[ref[i] for ref in targets] for i in range(len(targets[0]))]
        tm_metrics = SacreBLEUScore(tokenize=tokenize)
        tm_metrics.update(metrics_targets, preds)
        metrics_score = metrics_score_fn(metrics_targets, preds, tokenize)
        assert metrics_score == original_score
