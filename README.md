# <p align='center'> KSig </p>
## <p align='center'> GPU-accelerated computation of the signature kernel </p>
This is a [scikit-learn](https://github.com/scikit-learn/scikit-learn) compatible Python package for GPU-accelerated computation of the [signature kernel](https://jmlr.org/papers/v20/16-314.html) using [CuPy](https://github.com/cupy/cupy). This is a companion package to [GPSig](https://github.com/tgcsaba/GPSig), which implements (automatic) differentiable computations of the signature kernel using TensorFlow and GPFlow. 

## Introduction

The signature kernel is a mathematical construction that lifts a kernel on a given domain into **a kernel for sequences in that domain** with many theoretical and practical benefits:
- powerful theoretical guarantees, such as *universality* and optionality for *parametrization (warping) invariance*,
- generalizes classical sequence kernels, which therefore arise as special cases,
- strong performance on applied benchmark datasesets.

The signature kernel between sequences can be computed using an instantiation of the `ksig.kernels.SignatureKernel` class by lifting a kernel for vector-valued data (in this case an RBF from the `ksig.embeddings.kernels.RBFKernel` class) to a kernel for sequences as:
```python
import ksig

n_levels = 5 
# number of signature levels to use

base_kernel = ksig.embeddings.kernels.RBFKernel() 
# an RBF base kernel for vector-valued data which is lifted to a kernel for sequences

sig_kernel = ksig.kernels.SignatureKernel(n_levels, base_kernel=base_kernel) 
# a SignatureKernel object, which works as a callable for computing the signature kernel matrix

n_seq, l_seq, n_feat = 10, 50, 5 
X = np.random.randn(n_seq, l_seq, n_feat)
# generate 10 sequences of length 50 with 5 features

K_XX = sig_kernel(X) 
# compute the signature kernel matrix k(X, X)
# more efficient than calling sig_kernel(X, X)
# K_XX has shape (10, 10)

n_seq2, l_seq2 = 8, 20
Y = np.random.randn(n_seq2, l_seq2, n_feat)
# generate another array of 8 sequences of length 20 and 5 features

K_XY = sig_kernel(X, Y)
# compute the kernel matrix between arrays X and Y
# K_XY has shape (10, 8)


```

## Installation
Ideally in a clean [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with Python >= 3.6, run the command
```
pip install git+https://github.com/tgcsaba/ksig.git
```
  
## More Examples

Coming soon...


## Literature

Some papers that involve signature kernels in one way or another are (in chronological order):

- [Kernels for Sequentially Ordered Data](https://jmlr.org/papers/v20/16-314.html) by Kiraly, Oberhauser

- [Persistence Paths and Signature Features in Topological Data Analysis](https://arxiv.org/abs/1806.00381) by Chevyrev, Nanda, Oberhauser

- [Bayesian Learning From Sequential Data Using Gaussian Processes With Signature Covariances](http://proceedings.mlr.press/v119/toth20a.html) by Toth, Oberhauser

- [Signature Moments to Characterize Laws of Stochastic Processes](https://arxiv.org/abs/1810.10971) by Chevyrev, Oberhauser

- [A Data-Driven Market Simulator for Small Data Environments](https://arxiv.org/abs/2006.14498) by Bühler, Horvath, Lyons, Arribas, Wood

- [Path Signatures and Lie Groups](https://arxiv.org/abs/2007.06633) by Ghrist, Lee

- [Distribution Regression for Sequential Data](https://arxiv.org/abs/2006.05805) by Lemercier, Salvi, Damoulas, Bonilla, Lyons

- [The Signature Kernel is the Solution of a Goursat Problem](https://arxiv.org/abs/2006.14794) by Salvi, Cass, Foster, Lyons, Yang

- [Time-warping invariants of multidimensional time series](https://arxiv.org/abs/1906.05823) by Diehl, Ebrahimi-Fard, Tapia 

- [Generalized iterated-sums signatures](https://arxiv.org/abs/2012.04597) by Diehl, Ebrahimi-Fard, Tapia 


Please don't hesitate to contact us if your paper is missing here. 
  
## Feedback

For any queries or feedback, you can reach us at `csaba.toth@maths.ox.ac.uk` and `harald.oberhauser@maths.ox.ac.uk`.
  
