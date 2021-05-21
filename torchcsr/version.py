__version__ = '0.0.1+6387896'
git_version = '6387896aee96f4ae31f1c46c2244dac37d34858b'
from torchcsr import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
