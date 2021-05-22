### Build

Run on Linux, Volta GPU, recent PyTorch nightly

```
python setup.py clean && TORCH_CUDA_ARCH_LIST=Volta python setup.py develop
```

### Run softmax

Run
```
python run.py
```

which contains a padded+masked implementation of softmax and a call into sparse_softmax.

### TODO

 - Sparse CUDA kernel is registered at CPU with to/from device conversion. SparseCsrCUDA has no constructor, but we can add coverage out of tree here.

 - Need to setup row_offsets correctly in call to sputnik sparse_softmax

