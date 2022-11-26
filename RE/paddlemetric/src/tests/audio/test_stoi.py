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
from functools import partial

import pytest
import paddleext.torchapi as B
from pystoi import stoi as stoi_backend
from paddleext.torchapi import Tensor

from tests.helpers import seed_all
from tests.helpers.testers import MetricTester
from paddlemetrics.audio import STOI
from paddlemetrics.functional import stoi
from paddlemetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])

inputs_8k = Input(
    preds=B.rand(2, 3, 8000),
    target=B.rand(2, 3, 8000),
)
inputs_16k = Input(
    preds=B.rand(2, 3, 16000),
    target=B.rand(2, 3, 16000),
)


def stoi_original_batch(preds: Tensor, target: Tensor, fs: int, extended: bool):
    # shape: preds [BATCH_SIZE, Time] , target [BATCH_SIZE, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, Time] , target [NUM_BATCHES*BATCH_SIZE, Time]
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for b in range(preds.shape[0]):
        pesq_val = stoi_backend(target[b, ...], preds[b, ...], fs, extended)
        mss.append(pesq_val)
    return B.tensor(mss)


def average_metric(preds, target, metric_func):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


stoi_original_batch_8k_ext = partial(stoi_original_batch, fs=8000, extended=True)
stoi_original_batch_16k_ext = partial(stoi_original_batch, fs=16000, extended=True)
stoi_original_batch_8k_noext = partial(stoi_original_batch, fs=8000, extended=False)
stoi_original_batch_16k_noext = partial(stoi_original_batch, fs=16000, extended=False)


@pytest.mark.parametrize(
    "preds, target, sk_metric, fs, extended",
    [
        (inputs_8k.preds, inputs_8k.target, stoi_original_batch_8k_ext, 8000, True),
        (inputs_16k.preds, inputs_16k.target, stoi_original_batch_16k_ext, 16000, True),
        (inputs_8k.preds, inputs_8k.target, stoi_original_batch_8k_noext, 8000, False),
        (inputs_16k.preds, inputs_16k.target, stoi_original_batch_16k_noext, 16000, False),
    ],
)
class TestSTOI(MetricTester):
    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_stoi(self, preds, target, sk_metric, fs, extended, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            STOI,
            sk_metric=partial(average_metric, metric_func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
            metric_args=dict(fs=fs, extended=extended),
        )

    def test_stoi_functional(self, preds, target, sk_metric, fs, extended):
        self.run_functional_metric_test(
            preds,
            target,
            stoi,
            sk_metric,
            metric_args=dict(fs=fs, extended=extended),
        )

    def test_stoi_differentiability(self, preds, target, sk_metric, fs, extended):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=STOI,
            metric_functional=stoi,
            metric_args=dict(fs=fs, extended=extended),
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason="half support of core operations on not support before pytorch v1.6"
    )
    def test_stoi_half_cpu(self, preds, target, sk_metric, fs, extended):
        pytest.xfail("STOI metric does not support cpu + half precision")

    @pytest.mark.skipif(not B.cuda.is_available(), reason="test requires cuda")
    def test_stoi_half_gpu(self, preds, target, sk_metric, fs, extended):
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=STOI,
            metric_functional=partial(stoi, fs=fs, extended=extended),
            metric_args=dict(fs=fs, extended=extended),
        )


def test_error_on_different_shape(metric_class=STOI):
    metric = metric_class(16000)
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(B.randn(100), B.randn(50))


def test_on_real_audio():
    import os

    from scipy.io import wavfile

    current_file_dir = os.path.dirname(__file__)

    rate, ref = wavfile.read(os.path.join(current_file_dir, "examples/audio_speech.wav"))
    rate, deg = wavfile.read(os.path.join(current_file_dir, "examples/audio_speech_bab_0dB.wav"))
    assert B.allclose(
        stoi(B.from_numpy(deg), B.from_numpy(ref), rate).float(),
        B.tensor(0.6739177),
        rtol=0.0001,
        atol=1e-4,
    )
