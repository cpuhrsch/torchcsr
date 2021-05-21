__version__ = '0.0.1+22f6c37'
git_version = '22f6c3748647e9ad68899e06700a96675c898ec8'
from torchcsr import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
