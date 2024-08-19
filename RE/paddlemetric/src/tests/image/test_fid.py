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
from scipy.linalg import sqrtm as scipy_sqrtm
from B.utils.data import Dataset

from paddlemetrics.image.fid import FID, sqrtm
from paddlemetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE

B.manual_seed(42)


@pytest.mark.parametrize("matrix_size", [2, 10, 100, 500])
def test_matrix_sqrt(matrix_size):
    """test that metrix sqrt function works as expected."""

    def generate_cov(n):
        data = B.randn(2 * n, n)
        return (data - data.mean(dim=0)).T @ (data - data.mean(dim=0))

    cov1 = generate_cov(matrix_size)
    cov2 = generate_cov(matrix_size)

    scipy_res = scipy_sqrtm((cov1 @ cov2).numpy()).real
    tm_res = sqrtm(cov1 @ cov2)
    assert B.allclose(B.tensor(scipy_res).float(), tm_res, atol=1e-3)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_no_train():
    """Assert that metric never leaves evaluation mode."""

    class MyModel(B.nn.Module):
        def __init__(self):
            super().__init__()
            self.metric = FID()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert not model.metric.inception.training, "FID metric was changed to training mode which should not happen"


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_fid_pickle():
    """Assert that we can initialize the metric and pickle it."""
    metric = FID()
    assert metric

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)


def test_fid_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    with pytest.warns(
        UserWarning,
        match="Metric `FID` will save all extracted features in buffer."
        " For large datasets this may lead to large memory footprint.",
    ):
        _ = FID()

    if _TORCH_FIDELITY_AVAILABLE:
        with pytest.raises(ValueError, match="Integer input to argument `feature` must be one of .*"):
            _ = FID(feature=2)
    else:
        with pytest.raises(
            ValueError,
            match="FID metric requires that Torch-fidelity is installed."
            "Either install as `pip install paddlemetrics[image-quality]`"
            " or `pip install torch-fidelity`",
        ):
            _ = FID()

    with pytest.raises(TypeError, match="Got unknown input to argument `feature`"):
        _ = FID(feature=[1, 2])


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
@pytest.mark.parametrize("feature", [64, 192, 768, 2048])
def test_fid_same_input(feature):
    """if real and fake are update on the same data the fid score should be
    0."""
    metric = FID(feature=feature)

    for _ in range(2):
        img = B.randint(0, 255, (10, 3, 299, 299), dtype=B.uint8)
        metric.update(img, real=True)
        metric.update(img, real=False)

    assert B.allclose(B.cat(metric.real_features, dim=0), B.cat(metric.fake_features, dim=0))

    val = metric.compute()
    assert B.allclose(val, B.zeros_like(val), atol=1e-3)


class _ImgDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return self.imgs.shape[0]


@pytest.mark.skipif(not B.cuda.is_available(), reason="test is too slow without gpu")
@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch-fidelity")
def test_compare_fid(tmpdir, feature=2048):
    """check that the hole pipeline give the same result as torch-fidelity."""
    from torch_fidelity import calculate_metrics

    metric = FID(feature=feature).cuda()

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
        fid=True,
        feature_layer_fid=str(feature),
        batch_size=batch_size,
        save_cpu_ram=True,
    )

    tm_res = metric.compute()

    assert B.allclose(tm_res.cpu(), B.tensor([torch_fid["frechet_inception_distance"]]), atol=1e-3)
