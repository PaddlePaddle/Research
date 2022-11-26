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
from collections import namedtuple

import paddleext.torchapi as B

from tests.helpers.testers import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES

Input = namedtuple("InputMultiple", ["indexes", "preds", "target"])

# correct
_input_retrieval_scores = Input(
    indexes=B.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=B.rand(NUM_BATCHES, BATCH_SIZE),
    target=B.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_extra = Input(
    indexes=B.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
    preds=B.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=B.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
)

_input_retrieval_scores_int_target = Input(
    indexes=B.randint(high=10, size=(NUM_BATCHES, 2 * BATCH_SIZE)),
    preds=B.rand(NUM_BATCHES, 2 * BATCH_SIZE),
    target=B.randint(low=-1, high=4, size=(NUM_BATCHES, 2 * BATCH_SIZE)),
)

_input_retrieval_scores_float_target = Input(
    indexes=B.randint(high=10, size=(NUM_BATCHES, 2 * BATCH_SIZE)),
    preds=B.rand(NUM_BATCHES, 2 * BATCH_SIZE),
    target=B.rand(NUM_BATCHES, 2 * BATCH_SIZE),
)

# with errors
_input_retrieval_scores_no_target = Input(
    indexes=B.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=B.rand(NUM_BATCHES, BATCH_SIZE),
    target=B.randint(high=1, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_all_target = Input(
    indexes=B.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=B.rand(NUM_BATCHES, BATCH_SIZE),
    target=B.randint(low=1, high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_empty = Input(
    indexes=B.randint(high=10, size=[0]),
    preds=B.rand(0),
    target=B.randint(high=2, size=[0]),
)

_input_retrieval_scores_mismatching_sizes = Input(
    indexes=B.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE - 2)),
    preds=B.rand(NUM_BATCHES, BATCH_SIZE),
    target=B.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_mismatching_sizes_func = Input(
    indexes=B.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=B.rand(NUM_BATCHES, BATCH_SIZE - 2),
    target=B.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_wrong_targets = Input(
    indexes=B.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=B.rand(NUM_BATCHES, BATCH_SIZE),
    target=B.randint(low=-(2 ** 31), high=2 ** 31, size=(NUM_BATCHES, BATCH_SIZE)),
)
