__version__ = '0.0.1+f380204'
git_version = 'f380204ba6996aa26db0cbf426c463133b8b7caf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
