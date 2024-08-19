import functools
import re

import numpy as np
import pytest

from tests.classification.inputs import _input_binary_prob
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all

# TODO: replace this with official sklearn implementation after next sklearn release
from tests.helpers.non_sklearn_metrics import calibration_error as sk_calib
from tests.helpers.testers import THRESHOLD, MetricTester
from paddlemetrics import CalibrationError
from paddlemetrics.functional import calibration_error
from paddlemetrics.utilities.checks import _input_format_classification
from paddlemetrics.utilities.enums import DataType

seed_all(42)


def _sk_calibration(preds, target, n_bins, norm, debias=False):
    _, _, mode = _input_format_classification(preds, target, threshold=THRESHOLD)
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if mode == DataType.MULTICLASS:
        # binary label is whether or not the predicted class is correct
        sk_target = np.equal(np.argmax(sk_preds, axis=1), sk_target)
        sk_preds = np.max(sk_preds, axis=1)
    elif mode == DataType.MULTIDIM_MULTICLASS:
        # reshape from shape (N, C, ...) to (N*EXTRA_DIMS, C)
        sk_preds = np.transpose(sk_preds, axes=(0, 2, 1))
        sk_preds = sk_preds.reshape(np.prod(sk_preds.shape[:-1]), sk_preds.shape[-1])
        # reshape from shape (N, ...) to (N*EXTRA_DIMS,)
        # binary label is whether or not the predicted class is correct
        sk_target = np.equal(np.argmax(sk_preds, axis=1), sk_target.flatten())
        sk_preds = np.max(sk_preds, axis=1)
    return sk_calib(y_true=sk_target, y_prob=sk_preds, norm=norm, n_bins=n_bins, reduce_bias=debias)


@pytest.mark.parametrize("n_bins", [10, 15, 20])
@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_binary_prob.preds, _input_binary_prob.target),
        (_input_mcls_prob.preds, _input_mcls_prob.target),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target),
    ],
)
class TestCE(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_ce(self, preds, target, n_bins, ddp, dist_sync_on_step, norm):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=CalibrationError,
            sk_metric=functools.partial(_sk_calibration, n_bins=n_bins, norm=norm),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"n_bins": n_bins, "norm": norm},
        )

    def test_ce_functional(self, preds, target, n_bins, norm):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=calibration_error,
            sk_metric=functools.partial(_sk_calibration, n_bins=n_bins, norm=norm),
            metric_args={"n_bins": n_bins, "norm": norm},
        )


@pytest.mark.parametrize("preds, targets", [(_input_mlb_prob.preds, _input_mlb_prob.target)])
def test_invalid_input(preds, targets):
    for p, t in zip(preds, targets):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Calibration error is not well-defined for data with size {p.size()} and targets {t.size()}."
            ),
        ):
            calibration_error(p, t)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_binary_prob.preds, _input_binary_prob.target),
        (_input_mcls_prob.preds, _input_mcls_prob.target),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target),
    ],
)
def test_invalid_norm(preds, target):
    with pytest.raises(ValueError, match="Norm l3 is not supported. Please select from l1, l2, or max. "):
        calibration_error(preds, target, norm="l3")


@pytest.mark.parametrize("n_bins", [-10, -1, "fsd"])
@pytest.mark.parametrize(
    "preds, targets",
    [
        (_input_binary_prob.preds, _input_binary_prob.target),
        (_input_mcls_prob.preds, _input_mcls_prob.target),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target),
    ],
)
def test_invalid_bins(preds, targets, n_bins):
    for p, t in zip(preds, targets):
        with pytest.raises(ValueError, match=f"Expected argument `n_bins` to be a int larger than 0 but got {n_bins}"):
            calibration_error(p, t, n_bins=n_bins)
