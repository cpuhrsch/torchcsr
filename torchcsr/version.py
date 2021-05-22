__version__ = '0.0.1+442b092'
git_version = '442b092f103144ff3c3af170dff2c683081ea1ea'
from torchcsr import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
