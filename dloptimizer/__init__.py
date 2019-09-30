import pkg_resources
_dists = [d.key for d in pkg_resources.working_set]
if 'chainer' in _dists:
    from dloptimizer import chainer
if 'torch' in _dists:
    from dloptimizer import pytorch
if [s for s in _dists if 'tensorflow' in s]:
    from dloptimizer import tensorflow