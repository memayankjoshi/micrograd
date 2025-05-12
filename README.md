# Micrograd
Building an Autograd engine Micrograd (by [Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1))

### Micrograd:
- Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API

### Example usage:
```python 
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Micrograd.ipynb
- This file implements the **Value** class and **MLP** class.
- **Value** class contains all the basic Mathematical operations and backward pass for these operations.
- **MLP** class implements Multi Layer Perceptron using Layers and Neuron class.

### Micrograd_example.ipynb
- This file contain sample example to use Micrograd.
- Also implemented MLP.
