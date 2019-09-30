# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""Built-in optimizer classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.optimizers.Adastand')
class Adastand(Optimizer):
  """Adastand optimizer.

  Default parameters follow those provided in the original paper.

  Arguments:
      lr: float >= 0. Learning rate.
      beta_1: float, 0 < beta < 1. Generally close to 1.
      beta_2: float, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
      amsgrad: boolean. Whether to apply the AMSGrad variant of this
          algorithm from the paper "On the Convergence of Adam and
          Beyond".

  """

  def __init__(self,
               lr=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=None,
               decay=0.,
               amsgrad=False,
               **kwargs):
    super(Adastand, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.lr = K.variable(lr, name='lr')
      self.beta_1 = K.variable(beta_1, name='beta_1')
      self.beta_2 = K.variable(beta_2, name='beta_2')
      self.decay = K.variable(decay, name='decay')
    if epsilon is None:
      epsilon = K.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay
    self.amsgrad = amsgrad

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    t = math_ops.cast(self.iterations, K.floatx()) + 1
    lr_t = lr * (
        K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
        (1. - math_ops.pow(self.beta_1, t)))

    ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    if self.amsgrad:
      vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    else:
      vhats = [K.zeros(1) for _ in params]
    self.weights = [self.iterations] + ms + vs + vhats

    for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
      v_t = self.beta_2 * v + (1. - self.beta_2) * self.beta_2 * math_ops.square(g - m)
      m_t = (2. * self.beta_1 - 1) * m + (1. - self.beta_1) * g
      if self.amsgrad:
        vhat_t = math_ops.maximum(vhat, v_t)
        p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
        self.updates.append(state_ops.assign(vhat, vhat_t))
      else:
        p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

      self.updates.append(state_ops.assign(m, m_t))
      self.updates.append(state_ops.assign(v, v_t))
      new_p = p_t

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'beta_1': float(K.get_value(self.beta_1)),
        'beta_2': float(K.get_value(self.beta_2)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad
    }
    base_config = super(Adastand, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
