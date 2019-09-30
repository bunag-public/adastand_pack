# coding: UTF-8

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

"""SDProp for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export

@tf_export("train.SDPropOptimizer")
class SDPropOptimizer(optimizer.Optimizer):
  """Optimizer that implements the SDProp algorithm.

  """

  def __init__(self, rho=0.001, gamma=0.99, epsilon=1e-8,
               use_locking=False, name="SDProp"):
    """Construct a new SDProp optimizer.

    Initialization:

    ```
    m_0 <- 0 (Initialize initial 1st moment vector)
    v_0 <- 0 (Initialize initial 2nd moment vector)
    t <- 0 (Initialize timestep)
    ```

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section2 of the paper:

    ```
    t <- t + 1
    lr_t <- rho * sqrt(1 - gamma^t) / (1 - gamma^t)

    m_t <- gamma * m_{t-1} + (1 - gamma) * g_t
    v_t <- gamma * v_{t-1} + gamma * (1 - gamma) * (g_t - m_{t-1}) * (g_t - m_{t-1})
    variable <- variable - lr_t * g_t / (sqrt(v_t) + epsilon)
    ```

    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (gamma) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).

    Args:
      rho: A Tensor or a floating point value.  The learning rate.
      gamma: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      epsilon: A small constant for numerical stability.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "SDProp".
    """
    super(SDPropOptimizer, self).__init__(use_locking, name)
    self._lr = rho
    self._gamma = gamma
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._gamma_t = None
    self._epsilon_t = None

    # Variables to accumulate the powers of the gamma parameters.
    # Created in _create_slots when we know the variables to optimize.
    self._gamma_power = None

    # Created in SparseApply if needed.
    self._updated_lr = None

  def _get_gamma_accumulators(self):
    return self._gamma_power

  def _create_slots(self, var_list):
    # Create the gamma accumulators on the same device as the first
    # variable.
    if (self._gamma_power is None or
        self._gamma_power.graph is not var_list[0].graph):
      with ops.colocate_with(var_list[0]):
        self._gamma_power = variables.Variable(self._gamma,
                                               name="gamma_power",
                                               trainable=False)
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr, name="rho")
    self._gamma_t = ops.convert_to_tensor(self._gamma, name="gamma")
    self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.apply_sd_prop(
        var, m, v,
        math_ops.cast(self._gamma_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._gamma_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad, use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.resource_apply_sd_prop(
        var.handle, m.handle, v.handle,
        math_ops.cast(self._gamma_power, grad.dtype.base_dtype),
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._gamma_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    gamma_power = math_ops.cast(self._gamma_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = lr_t * math_ops.sqrt(1 - gamma_power)
    # v_t = gamma * v + gamma * (1 - gamma) * ((g_t - m) * (g_t - m))
    v = self.get_slot(var, "v")
    m = self.get_slot(var, "m")
    v_scaled_g_values = gamma_t * (1 - gamma_t)
    m_t1 = state_ops.assign(m, -m, use_locking=self._use_locking)
    m_t1 = state_ops.scatter_add(m_t1, grad.indices, grad.values, use_locking=self._use_locking)
    m_t1 = state_ops.assign(m_t1, m_t1 * m_t1, use_locking=self._use_locking)
    m_t1 = state_ops.assign(m_t1, v_scaled_g_values * m_t1 , use_locking=self._use_locking)
    v_t = state_ops.assign(v, v * gamma_t, use_locking=self._use_locking)
    v_t = state_ops.assign_add(v_t, m_t1, use_locking=self._use_locking)
    
    # m_t = gamma * m + (1 - gamma) * g_t
    m_scaled_g_values = grad.values * (1 - gamma_t)
    m_t = state_ops.assign(m, m * gamma_t,
                           use_locking=self._use_locking)
    m_t = state_ops.scatter_add(m_t, grad.indices, m_scaled_g_values,
                               use_locking=self._use_locking)

    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(var,
                                      lr * m_t / (v_sqrt + epsilon_t),
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      with ops.colocate_with(self._gamma_power):
        update_gamma = self._gamma_power.assign(
            self._gamma_power * self._gamma_t,
            use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_gamma],
                                  name=name_scope)
