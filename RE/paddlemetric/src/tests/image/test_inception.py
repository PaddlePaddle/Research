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
import pickle

import pytest
import paddleext.torchapi as B
from B.utils.data import Dataset

from paddlemetrics.image.inception import IS
from paddlemetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE

B.manual_seed(42)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(B.nn.Module):
        def __init__(self):
            super().__init__()
            self.metric = IS()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert not model.metric.inception.training, "IS metric was changed to training mode which should not happen"


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_is_pickle():
    """Assert that we can initialize the metric and pickle it."""
    metric = IS()
    assert metric

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)


def test_is_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    with pytest.warns(
        UserWarning,
        match="Metric `IS` will save all extracted features in buffer."
        " For large datasets this may lead to large memory footprint.",
    ):
        IS()

    if _TORCH_FIDELITY_AVAILABLE:
        with pytest.raises(ValueError, match="Integer input to argument `feature` must be one of .*"):
            _ = IS(feature=2)
    else:
        with pytest.raises(
            ValueError,
            match="IS metric requires that Torch-fidelity is installed."
            "Either install as `pip install paddlemetrics[image-quality]`"
            " or `pip install torch-fidelity`",
        ):
            IS()

    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        IS(feature=[1, 2])


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_is_update_compute():
    metric = IS()

    for _ in range(2):
        img = B.randint(0, 255, (10, 3, 299, 299), dtype=B.uint8)
        metric.update(img)

    mean, std = metric.compute()
    assert mean >= 0.0
    assert std >= 0.0


class _ImgDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return self.imgs.shape[0]


@pytest.mark.skipif(not B.cuda.is_available(), reason="test is too slow without gpu")
@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_compare_is(tmpdir):
    """check that the hole pipeline give the same result as torch-fidelity."""
    from torch_fidelity import calculate_metrics

    metric = IS(splits=1).cuda()

    # Generate some synthetic data
    img1 = B.randint(0, 255, (100, 3, 299, 299), dtype=B.uint8)

    batch_size = 10
    for i in range(img1.shape[0] // batch_size):
        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda())

    torch_fid = calculate_metrics(
        input1=_ImgDataset(img1), isc=True, isc_splits=1, batch_size=batch_size, save_cpu_ram=True
    )

    tm_mean, _ = metric.compute()

    assert B.allclose(tm_mean.cpu(), B.tensor([torch_fid["inception_score_mean"]]), atol=1e-3)
