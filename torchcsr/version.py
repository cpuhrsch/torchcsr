__version__ = '0.0.1+f9a9a9f'
git_version = 'f9a9a9f7991474ed4be956f4991aa6cc3f2e992d'
from torchcsr import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
