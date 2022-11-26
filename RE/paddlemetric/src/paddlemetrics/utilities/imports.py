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
"""Import utilities."""
import operator
from importlib import import_module
from importlib.util import find_spec
from typing import Callable, Optional

from packaging.version import Version
from pkg_resources import DistributionNotFound, get_distribution


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False


def _compare_version(package: str, op: Callable, version: str) -> Optional[bool]:
    """Compare package version with some requirements.

    >>> import operator
    >>> _compare_version("torch", operator.ge, "0.1")
    True
    >>> _compare_version("any_module", operator.ge, "0.0")  # is None
    """
    if not _module_available(package):
        return None
    try:
        pkg = import_module(package)
        pkg_version = pkg.__version__  # type: ignore
    except (ModuleNotFoundError, DistributionNotFound):
        return None
    except AttributeError:
        pkg_version = get_distribution(package).version
    except ImportError:
        # catches cyclic imports - the case with integrated libs
        # see: https://stackoverflow.com/a/32965521
        pkg_version = get_distribution(package).version
    try:
        pkg_version = Version(pkg_version)
    except TypeError:
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, Version(version))


_TORCH_LOWER_1_4: Optional[bool] = False
_TORCH_LOWER_1_5: Optional[bool] = False
_TORCH_LOWER_1_6: Optional[bool] = False
_TORCH_GREATER_EQUAL_1_6: Optional[bool] = True
_TORCH_GREATER_EQUAL_1_7: Optional[bool] = True

_LIGHTNING_AVAILABLE: bool = False
_JIWER_AVAILABLE: bool = _module_available("jiwer")
_NLTK_AVAILABLE: bool = _module_available("nltk")
_ROUGE_SCORE_AVAILABLE: bool = _module_available("rouge_score")
_BERTSCORE_AVAILABLE: bool = _module_available("bert_score")
_SCIPY_AVAILABLE: bool = _module_available("scipy")
_TORCH_FIDELITY_AVAILABLE: bool = _module_available("torch_fidelity")
_LPIPS_AVAILABLE: bool = _module_available("lpips")
_TQDM_AVAILABLE: bool = _module_available("tqdm")
_TRANSFORMERS_AVAILABLE: bool = _module_available("transformers")
_PESQ_AVAILABLE: bool = _module_available("pesq")
_SACREBLEU_AVAILABLE: bool = _module_available("sacrebleu")
_REGEX_AVAILABLE: bool = _module_available("regex")
_PYSTOI_AVAILABLE: bool = _module_available("pystoi")
