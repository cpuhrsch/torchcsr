__version__ = '0.0.1+7f45b05'
git_version = '7f45b05940459648128abad8e939186aedf67b6e'
from torchcsr import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
