"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a, b: NDArray):
        return array_api.multiply(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return (EWiseMul()(out_grad, rhs), (EWiseMul()(out_grad, lhs)))


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):        
        return mul_scalar(self.scalar, out_grad)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return ((self.scalar) * power_scalar(a, self.scalar - 1) * out_grad,)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad * power_scalar(b, -1), out_grad * (-1) * a * power_scalar(b, -2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # return (out_grad / self.scalar,)
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (transpose(out_grad, self.axes),)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (reshape(out_grad, node.inputs[0].shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # notice the axes broadcased (repeated) would do extra work to contribute to the output
        # e.g. input tensor is shape (3, )
        # broadcased to shape (3, 4)
        # now if the op is summation
        # each of the original len-3 tensor is adding itself 4 times to the output
        # thus the gradient = out_grad * some all-1 tensor, or
        # gradient = summation(out_grad) on the axes where broadcasting is taking place
        a = node.inputs[0]
        a_shape = a.shape
        axes = []
        if a_shape == () or a_shape == (1,): #input is a scalar
            axes = tuple(range(len(self.shape)))
        else: # input is some tensor
            for i in range(len(a_shape)):
                if a_shape[i] != self.shape[i]:
                    axes.append(i)
            axes = tuple(axes)
        return (reshape(summation(out_grad, axes), a_shape),)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # summation for a tensor would have gradient=1
        # the idea is, if input tensor is shape (5, 4) and the summation sums it to shape (5, )
        # the out_grad would be shape (5, )
        # now we just need to multiple the out_grad with the all-1 tensor
        # or a shape (5, 4) tensor with each column the same
        if self.axes is None:
            return broadcast_to(out_grad, node.inputs[0].shape)
        a = node.inputs[0]
        a_shape = a.shape # (a1,a2,a3...aD)
        to_shape = [i for i in a_shape]
        for i in range(len(a_shape)):
            if i in self.axes:
                to_shape[i] = 1
        to_shape = tuple(to_shape) #(a1,1,a3...1)
        return (broadcast_to(reshape(out_grad, to_shape), a_shape),)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs 
        # key is that the shape of a gradient to something is the same as the shape of that something
        a_grad, b_grad = matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        if len(a.shape) < len(a_grad.shape):
            a_grad = a_grad.sum(tuple([i for i in range(len(a_grad.shape) - len(a.shape))]))
        if len(b.shape) < len(b_grad.shape):
            b_grad = b_grad.sum(tuple([i for i in range(len(b_grad.shape) - len(b.shape))]))
        return a_grad, b_grad        
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-1 * out_grad,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return EWiseDiv()(out_grad, a)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return EWiseMul()(exp(a), out_grad)        
        # return out_grad * exp_1(node.inputs[0])
        ### END YOUR SOLUTION

def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        out[out > 0] = 1
        return EWiseMul()(out_grad, Tensor(out))
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
