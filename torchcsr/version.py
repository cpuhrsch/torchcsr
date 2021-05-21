__version__ = '0.0.1+6908bb7'
git_version = '6908bb76060690d252f08974682ee003eab1bdf3'
from torchcsr import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
