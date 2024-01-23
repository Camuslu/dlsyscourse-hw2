"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

from needle.init.init_initializers import *
from needle.ops.ops_mathematic import *
from needle.ops.ops_logarithmic import *


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = kaiming_uniform(self.in_features, self.out_features, dtype=dtype)
        self.bias = None
        if bias:
            self.bias = reshape(kaiming_uniform(self.out_features, 1, dtype=dtype), (1, self.out_features))
        
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # X is [batch_size, input_feature]
        b_size = X.shape[0]
        if self.bias:
            bias_broadcast = broadcast_to(self.bias, (b_size, self.out_features))
            return matmul(X, self.weight) + bias_broadcast
        else:
            return matmul(X, self.weight)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # X is shape (B,X_0,X_1,...); change it to shape (B, X_0 * X_1 * ...)
        b_size = X.shape[0]
        flat_shape = 1
        for dim in X.shape[1:]:
            flat_shape *= dim
        return reshape(X, (b_size, flat_shape))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules # tuple

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module.forward(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # logits is [batch_size, ]
        # y is [batch_size]
        batch_size = logits.shape[0] 
        classes = logits.shape[1] 
        y_one_hot = one_hot(classes, y) # [batch_size, classes]
        log_sum_exp_logits = logsumexp(logits, axes=(1,)) # [batch_size]
        # print("Tianhao debug", log_sum_exp_logits.realize_cached_data)
        z_y = summation(EWiseMul()(logits, y_one_hot), axes=(1,)) # [batch_size]
        # print("Tianhao debug", z_y.realize_cached_data)
        loss = log_sum_exp_logits - z_y
        # print("Tianhao debug", loss.realize_cached_data)
        avg_loss = divide_scalar(summation(loss), batch_size)
        return avg_loss
        ### END YOUR SOLUTION

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = broadcast_to(ndl.Tensor(array_api.array([1.0]), dtype=dtype), shape=(self.dim,))
        self.bias = broadcast_to(ndl.Tensor(array_api.array([0.0]), dtype=dtype), shape=(self.dim,))
        self.running_mean = broadcast_to(ndl.Tensor(array_api.array([0.0]), dtype=dtype), shape=(self.dim,))
        self.running_var = broadcast_to(ndl.Tensor(array_api.array([1.0]), dtype=dtype), shape=(self.dim,))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            b_size = x.shape[0]
            mean = reshape(summation(x, axes=(0,)) / b_size, shape=(self.dim,)) # row-wise; [dim]
            x_minus_mean = x - broadcast_to(mean, shape=(b_size, self.dim)) # [batch_size, dim]
            x_minus_mean_sq = power_scalar(x_minus_mean, 2) # [batch_size, dim]
            var_x = reshape(summation(x_minus_mean_sq, axes=(0,)) / b_size, shape=(self.dim,)) # [dim,]
            var_x_plus_eps = add_scalar(var_x, self.eps) # [dim]
            std = reshape(power_scalar(var_x_plus_eps, 0.5), shape=(1, self.dim)) # [1, dim]
            std_broadcast = broadcast_to(std, shape=(b_size, self.dim)) # [batch_size, dim]
            normalized = EWiseDiv()(x_minus_mean, std_broadcast)
            normalized_with_weight = EWiseMul()(normalized, 
                                            broadcast_to(self.weight, shape=(b_size, self.dim)))
            normalized_with_weight_plus_bias = EWiseAdd()(normalized_with_weight,
                                                      broadcast_to(self.bias, shape=(b_size, self.dim)))
            self.update_running_mean_var(new_mean = mean, new_var = var_x)
            return normalized_with_weight_plus_bias
        else:
            x_minus_mean = x - broadcast_to(self.running_mean, shape=(b_size, self.dim)) # [batch_size, dim]
            var_x_plus_eps = add_scalar(self.running_var, self.eps) # [dim]
            std = reshape(power_scalar(var_x_plus_eps, 0.5), shape=(1, self.dim)) # [1, dim]
            std_broadcast = broadcast_to(std, shape=(b_size, self.dim)) # [batch_size, dim]
            normalized = EWiseDiv()(x_minus_mean, std_broadcast)
            normalized_with_weight = EWiseMul()(normalized, 
                                            broadcast_to(self.weight, shape=(b_size, self.dim)))
            normalized_with_weight_plus_bias = EWiseAdd()(normalized_with_weight,
                                                      broadcast_to(self.bias, shape=(b_size, self.dim)))
            return normalized_with_weight_plus_bias    

        ### END YOUR SOLUTION

    def update_running_mean_var(self, new_mean: Tensor, new_var: Tensor):
        # new_mean and new_var are both of shape (dim,)
        self.running_mean = EWiseAdd()((1 - self.momentum) * self.running_mean, new_mean * self.momentum)
        self.running_var = EWiseAdd()((1 - self.momentum) * self.running_var, new_var * self.momentum)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = broadcast_to(ndl.Tensor(array_api.array([1.0]), dtype=dtype), shape=(self.dim,))
        self.bias = broadcast_to(ndl.Tensor(array_api.array([0.0]), dtype=dtype), shape=(self.dim,))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        b_size = x.shape[0]
        mean = reshape(summation(x, axes=(1,)) / self.dim, shape=(b_size, 1)) # row-wise; [batch_size]
        print("Tianhao debug", mean.realize_cached_data)
        x_minus_mean = x - broadcast_to(mean, shape=(b_size, self.dim)) # [batch_size, dim]
        print("Tianhao debug", x_minus_mean.realize_cached_data)
        x_minus_mean_sq = power_scalar(x_minus_mean, 2) # [batch_size, dim]
        print("Tianhao debug", x_minus_mean_sq.realize_cached_data)
        var_x = summation(x_minus_mean_sq, axes=(1,)) / self.dim # [batch_size]
        print("Tianhao debug", var_x.realize_cached_data)
        var_x_plus_eps = add_scalar(var_x, self.eps) # [batch_size]
        print("Tianhao debug", var_x_plus_eps.realize_cached_data)
        std = reshape(power_scalar(var_x_plus_eps, 0.5), shape=(b_size, 1)) # [batch_size, 1]
        print("Tianhao debug", std.realize_cached_data)
        std_broadcast = broadcast_to(std, shape=(b_size, self.dim)) # [batch_size, dim]
        print("Tianhao debug", std_broadcast.realize_cached_data)
        normalized = EWiseDiv()(x_minus_mean, std_broadcast)
        print("Tianhao debug", normalized.realize_cached_data)
        normalized_with_weight = EWiseMul()(normalized, 
                                            broadcast_to(self.weight, shape=(b_size, self.dim)))
        print("Tianhao debug", normalized_with_weight.realize_cached_data)
        normalized_with_weight_plus_bias = EWiseAdd()(normalized_with_weight,
                                                      broadcast_to(self.bias, shape=(b_size, self.dim)))
        print("Tianhao debug", normalized_with_weight_plus_bias.realize_cached_data)
        return normalized_with_weight_plus_bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training is False:
            return x # no dropout in effect
        retain = randb(*x.shape, p = 1.0 - self.p) # the 1 - self.p is the "retain rate"
        return EWiseMul()(x, retain) / (1.0 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
