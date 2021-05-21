__version__ = '0.0.1+3c0a558'
git_version = '3c0a55886df6799ee5c947670102c506c704184d'
from torchcsr import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
