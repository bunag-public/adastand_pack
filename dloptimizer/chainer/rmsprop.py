import numpy

from chainer.backends import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.alpha = 0.99
_default_hyperparam.eps = 1e-8
_default_hyperparam.sdprop = False


class RMSpropRule(optimizer.UpdateRule):

    """Update rule for RMSprop.

    See :class:`~chainer.optimizers.RMSprop` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        alpha (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, parent_hyperparam=None, lr=None, alpha=None, eps=None, sdprop=None):
        super(RMSpropRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if eps is not None:
            self.hyperparam.eps = eps
        if sdprop is not None:
            self.hyperparam.sdprop = sdprop

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['ms'] = xp.zeros_like(param.data)
            if self.hyperparam.sdprop:
                self.state['m'] = xp.zeros_like(param.data)


    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of RMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        ms = self.state['ms']

        if hp.sdprop:
            m = self.state['m']
            ms += (1 - hp.alpha) * (hp.alpha * (grad - m) * (grad - m) - ms)
            m += (1 - hp.alhpha) * (grad - m)
        else:
            ms += (1 - hp.alpha) * (grad * grad - ms)
        param.data -= hp.lr * grad / (numpy.sqrt(ms) + eps)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if eps == 0:
            raise ValueError(
                'eps of RMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))

        if hp.sdprop:
            cuda.elementwise(
                'T grad, T lr, T alpha, T one_minus_alpha, T eps',
                'T param, T ms, T m',
                '''T mnorm = grad - m;
                   ms += one_minus_alpha * (alpha * mnorm * mnorm - ms);
                   m += one_minus_alpha * mnorm;
                   param -= lr * grad / (sqrt(ms) + eps);''',
                'sdprop')(grad, hp.lr, hp.alpha, 1- hp.alpha,
                           eps, param.data, self.state['ms'], self.state['m'])
        else:
            cuda.elementwise(
                'T grad, T lr, T alpha, T eps',
                'T param, T ms',
                '''ms = alpha * ms + (1 - alpha) * grad * grad;
                   param -= lr * grad / (sqrt(ms) + eps);''',
                'rmsprop')(grad, hp.lr, hp.alpha,
                           eps, param.data, self.state['ms'])


class RMSprop(optimizer.GradientMethod):

    """RMSprop optimizer.

    See: T. Tieleman and G. Hinton (2012). Lecture 6.5 - rmsprop, COURSERA:
    Neural Networks for Machine Learning.

    Args:
        lr (float): Learning rate.
        alpha (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, lr=_default_hyperparam.lr, alpha=_default_hyperparam.alpha,
                 eps=_default_hyperparam.eps, sdprop=_default_hyperparam.sdprop):
        super(RMSprop, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.alpha = alpha
        self.hyperparam.eps = eps
        self.hyperparam.sdprop = sdprop

    lr = optimizer.HyperparameterProxy('lr')
    alpha = optimizer.HyperparameterProxy('alpha')
    eps = optimizer.HyperparameterProxy('eps')
    sdprop = optimizer.HyperparameterProxy('sdprop')

    def create_update_rule(self):
        return RMSpropRule(self.hyperparam)
