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

from paddlemetrics.image.kid import KID
from paddlemetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE

B.manual_seed(42)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(B.nn.Module):
        def __init__(self):
            super().__init__()
            self.metric = KID()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert not model.metric.inception.training, "FID metric was changed to training mode which should not happen"


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_kid_pickle():
    """Assert that we can initialize the metric and pickle it."""
    metric = KID()
    assert metric

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)


def test_kid_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    with pytest.warns(
        UserWarning,
        match="Metric `KID` will save all extracted features in buffer."
        " For large datasets this may lead to large memory footprint.",
    ):
        KID()

    if _TORCH_FIDELITY_AVAILABLE:
        with pytest.raises(ValueError, match="Integer input to argument `feature` must be one of .*"):
            KID(feature=2)
    else:
        with pytest.raises(
            ValueError,
            match="KID metric requires that Torch-fidelity is installed."
            "Either install as `pip install paddlemetrics[image]`"
            " or `pip install torch-fidelity`",
        ):
            KID()

    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        KID(feature=[1, 2])

    with pytest.raises(ValueError, match="Argument `subset_size` should be smaller than the number of samples"):
        m = KID()
        m.update(B.randint(0, 255, (5, 3, 299, 299), dtype=B.uint8), real=True)
        m.update(B.randint(0, 255, (5, 3, 299, 299), dtype=B.uint8), real=False)
        m.compute()


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_kid_extra_parameters():
    with pytest.raises(ValueError, match="Argument `subsets` expected to be integer larger than 0"):
        KID(subsets=-1)

    with pytest.raises(ValueError, match="Argument `subset_size` expected to be integer larger than 0"):
        KID(subset_size=-1)

    with pytest.raises(ValueError, match="Argument `degree` expected to be integer larger than 0"):
        KID(degree=-1)

    with pytest.raises(ValueError, match="Argument `gamma` expected to be `None` or float larger than 0"):
        KID(gamma=-1)

    with pytest.raises(ValueError, match="Argument `coef` expected to be float larger than 0"):
        KID(coef=-1)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
@pytest.mark.parametrize("feature", [64, 192, 768, 2048])
def test_kid_same_input(feature):
    """test that the metric works."""
    metric = KID(feature=feature, subsets=5, subset_size=2)

    for _ in range(2):
        img = B.randint(0, 255, (10, 3, 299, 299), dtype=B.uint8)
        metric.update(img, real=True)
        metric.update(img, real=False)

    assert B.allclose(B.cat(metric.real_features, dim=0), B.cat(metric.fake_features, dim=0))

    mean, std = metric.compute()
    assert mean != 0.0
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
def test_compare_kid(tmpdir, feature=2048):
    """check that the hole pipeline give the same result as torch-fidelity."""
    from torch_fidelity import calculate_metrics

    metric = KID(feature=feature, subsets=1, subset_size=100).cuda()

    # Generate some synthetic data
    img1 = B.randint(0, 180, (100, 3, 299, 299), dtype=B.uint8)
    img2 = B.randint(100, 255, (100, 3, 299, 299), dtype=B.uint8)

    batch_size = 10
    for i in range(img1.shape[0] // batch_size):
        metric.update(img1[batch_size * i : batch_size * (i + 1)].cuda(), real=True)

    for i in range(img2.shape[0] // batch_size):
        metric.update(img2[batch_size * i : batch_size * (i + 1)].cuda(), real=False)

    torch_fid = calculate_metrics(
        input1=_ImgDataset(img1),
        input2=_ImgDataset(img2),
        kid=True,
        feature_layer_fid=str(feature),
        batch_size=batch_size,
        kid_subsets=1,
        kid_subset_size=100,
        save_cpu_ram=True,
    )

    tm_mean, tm_std = metric.compute()

    assert B.allclose(tm_mean.cpu(), B.tensor([torch_fid["kernel_inception_distance_mean"]]), atol=1e-3)
    assert B.allclose(tm_std.cpu(), B.tensor([torch_fid["kernel_inception_distance_std"]]), atol=1e-3)
