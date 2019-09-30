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

@tf_export('keras.optimizers.SDprop')
class SDprop(Optimizer):
  """SDprop optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values
  (except the learning rate, which can be freely tuned).

  This optimizer is usually a good choice for recurrent
  neural networks.

  Arguments:
      lr: float >= 0. Learning rate.
      rho: float >= 0.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.

  """

  def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0., **kwargs):
    super(SDprop, self).__init__(**kwargs)
    with K.name_scope(self.__class__.__name__):
      self.lr = K.variable(lr, name='lr')
      self.rho = K.variable(rho, name='rho')
      self.decay = K.variable(decay, name='decay')
      self.iterations = K.variable(0, dtype='int64', name='iterations')
    if epsilon is None:
      epsilon = K.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

    self.weights = [ms] + vs
    self.updates = [state_ops.assign_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                K.dtype(self.decay))))

    for p, g, m, v in zip(params, grads, ms, vs):
      # update accumulator
      new_v = self.rho * v + (1. - self.rho) * self.rho * math_ops.square(g - m)
      new_m = self.rho * m + (1. - self.rho) * g
      self.updates.append(state_ops.assign(m, new_m))
      self.updates.append(state_ops.assign(v, new_v))
      new_p = p - lr * g / (K.sqrt(new_v) + self.epsilon)

      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)

      self.updates.append(state_ops.assign(p, new_p))
    return self.updates

  def get_config(self):
    config = {
        'lr': float(K.get_value(self.lr)),
        'rho': float(K.get_value(self.rho)),
        'decay': float(K.get_value(self.decay)),
        'epsilon': self.epsilon
    }
    base_config = super(SDprop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
