# <p align='center'> K<sub>Sig</sub> </p>
## <p align='center'> GPU-accelerated computation of the signature kernel </p>
A [scikit-learn](https://github.com/scikit-learn/scikit-learn) compatible Python package for GPU-accelerated computation of the [signature kernel](https://jmlr.org/papers/v20/16-314.html) using [CuPy](https://github.com/cupy/cupy). This is a companion package to [GPSig](https://github.com/tgcsaba/GPSig), which implements (automatic) differentiable computations of the signature kernel using TensorFlow and GPFlow. 

## Introduction

The signature kernel is a mathematical construction that lifts a kernel on a given domain to **a kernel for sequences in that domain** with many theoretical and practical benefits:
- powerful theoretical guarantees, such as *universality* and optionality for *parametrization (warping) invariance*,
- generalizes classical sequence kernels, which therefore arise as special cases,
- strong performance on applied benchmark datasesets.

The signature kernel between sequences can be computed using an instantiation of the `ksig.kernels.SignatureKernel` class by lifting a static kernel (i.e. in this case an RBF kernel for vector-valued data from the `ksig.static.kernels.RBFKernel` class) to a kernel for sequences as:
```python
import numpy as np
import ksig

n_levels = 5 
# number of signature levels to use

static_kernel = ksig.static.kernels.RBFKernel() 
# an RBF base kernel for vector-valued data which is lifted to a kernel for sequences

sig_kernel = ksig.kernels.SignatureKernel(n_levels, static_kernel=static_kernel) 
# a SignatureKernel object, which works as a callable for computing the signature kernel matrix

n_seq, l_seq, n_feat = 10, 50, 5 
X = np.random.randn(n_seq, l_seq, n_feat)
# generate 10 sequences of length 50 with 5 features

K_XX = sig_kernel(X) # K_XX has shape (10, 10)
# compute the signature kernel matrix k(X, X)
# more efficient than calling sig_kernel(X, X)

K_X = sig_kernel(X, diag=True) # K_X has shape (10,)
# compute only the diagonal entries of the signature kernel matrix

n_seq2, l_seq2 = 8, 20
Y = np.random.randn(n_seq2, l_seq2, n_feat)
# generate another array of 8 sequences of length 20 and 5 features

K_XY = sig_kernel(X, Y) # K_XY has shape (10, 8)
# compute the kernel matrix between arrays X and Y

```

## Installation
Ideally in a clean [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with Python >= 3.7, run the command
```
pip install git+https://github.com/tgcsaba/ksig.git
```
Make sure that the `CUDA_PATH` environment variable is properly set if CuPy is not installed yet on your system, otherwise the installer might not find the right location causing the installation to fail. The default CUDA installation path on Linux is `/usr/local/cuda/`, while on Windows it is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\${CUDA_VERSION}\` where `${CUDA_VERSION}` is for example `v10.0`.
  
## More Examples

### Scaling to large datasets
A computational bottleneck associated with the full-rank signature kernel is a joint quadratic complexity in the number of training examples (N) and the length of the sequences (L), i.e. O(N<sup>2</sup>&middot;L<sup>2</sup>). The low-rank signature kernel gets around this issue by [computing a low-rank approximation](https://jmlr.org/papers/volume20/16-314/16-314.pdf#page=29) to the signature kernel matrix.
This variant of the kernel variant can be defined using the `ksig.kernel.LowRankSignatureKernel` class, and its two main ingredients are
* a `static_features` object from `ksig.static.features` for low-rank approximation of a static kernel, such as:
    * `NystroemFeatures` implementing the [Nyström Method](https://papers.nips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf);
    * `RBFFourierFeatures` using [Random Fourier Features](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) for Monte Carlo approximation of the RBF kernel;
* a `projection` object from `ksig.projections` to keep the size of the low-rank signature factors manageable using randomized projections:
    * `GaussianRandomProjection` implementing vanilla [Gaussian Random Projections](https://arxiv.org/ftp/arxiv/papers/1301/1301.3849.pdf);
    * `VerySparseRandomProjection` using [Very Sparse Random Projections](https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf);
    * `CountSketchRandomProjection` for [CountSketch](https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarCF.pdf) with polynomial multiplication via FFT, i.e. the [TensorSketch](https://dl.acm.org/doi/10.1145/2487575.2487591);
    * `TensorizedRandomProjection` corresponding to [Tensorized Random Projections](https://proceedings.mlr.press/v108/rakhshan20a/rakhshan20a.pdf) in the CP format.

The following example can be computed on a GPU with about ~5Gb of free memory in a matter of seconds:
```python
import numpy as np
import ksig

n_levels = 5 
# number of signature levels to use

n_components = 100
# number of components to use in the static features and the randomized projections

static_kernel = ksig.static.kernels.RBFKernel() 
# an RBF kernel for vector-valued data

static_feat = ksig.static.features.NystroemFeatures(static_kernel, n_components=n_components)
# Nystroem features with an RBF base kernel

proj = ksig.projections.CountSketchRandomProjection(n_components=n_components)
# a CountSketch random projection 

lr_sig_kernel = ksig.kernels.LowRankSignatureKernel(n_levels=n_levels, static_features=static_feat, projection=proj)
# a low-rank signature kernel, which additionally to working as a callable for kernel matrix computations
# also implements a fit method, which must be used to fit the kernel (and its subobjects) to the data
# and a transform method, which can be used to transform an array of paths to their corresponding low-rank features

n_seq, l_seq, n_feat = 1000, 200, 100
X = np.random.randn(n_seq, l_seq, n_feat)
# generate 1000 sequences of length 200 with 100 features

lr_sig_kernel.fit(X)
# fit the kernel to the data

K_XX = lr_sig_kernel(X) # K_XX has shape (1000, 1000)
# compute the low-rank signature kernel matrix k(X, X)

n_seq2, l_seq2 = 800, 250
Y = np.random.randn(n_seq2, l_seq2, n_feat)
# generate another array of 800 sequences of length 250 and 100 features

K_XY = lr_sig_kernel(X, Y) # K_XY has shape (1000, 800)
# compute the kernel matrix between arrays X and Y
# the kernel does not have to be fitted a second time

P_X = lr_sig_kernel.transform(X) # P_X has shape (n_seq, 1+n_levels*n_components) i.e. (1000, 501) in this case
P_Y = lr_sig_kernel.transform(Y) # P_Y shape shape (800, 501)
# alternatively, one may directly compute the low-rank representations for both X and Y
# and then use these features to compute the kernel matrices K_XX and K_XY

print(np.linalg.norm(K_XX - P_X @ P_X.T)) # 1.5336806154787045e-14
print(np.linalg.norm(K_XY - P_X @ P_Y.T)) # 0.0
```

## Documentation
Coming soon...

## Literature

Some papers that involve signature kernels in one way or another are (in chronological order):

- [Kernels for Sequentially Ordered Data](https://jmlr.org/papers/v20/16-314.html) by Kiraly, Oberhauser

- [Persistence Paths and Signature Features in Topological Data Analysis](https://arxiv.org/abs/1806.00381) by Chevyrev, Nanda, Oberhauser

- [Bayesian Learning From Sequential Data Using GPs With Signature Covariances](http://proceedings.mlr.press/v119/toth20a.html) by Toth, Oberhauser

- [Signature Moments to Characterize Laws of Stochastic Processes](https://arxiv.org/abs/1810.10971) by Chevyrev, Oberhauser

- [A Data-Driven Market Simulator for Small Data Environments](https://arxiv.org/abs/2006.14498) by Bühler, Horvath, Lyons, Arribas, Wood

- [Path Signatures and Lie Groups](https://arxiv.org/abs/2007.06633) by Ghrist, Lee

- [Distribution Regression for Sequential Data](https://arxiv.org/abs/2006.05805) by Lemercier, Salvi, Damoulas, Bonilla, Lyons

- [The Signature Kernel is the Solution of a Goursat Problem](https://arxiv.org/abs/2006.14794) by Salvi, Cass, Foster, Lyons, Yang

- [Time-warping invariants of multidimensional time series](https://arxiv.org/abs/1906.05823) by Diehl, Ebrahimi-Fard, Tapia 

- [Generalized iterated-sums signatures](https://arxiv.org/abs/2012.04597) by Diehl, Ebrahimi-Fard, Tapia 

- [SigGPDE: Scaling Sparse Gaussian Processes on Sequential Data](https://arxiv.org/abs/2105.04211) by Lemercier, Salvi, Cass, Bonilla, Damoulas, Lyons


Please don't hesitate to contact us if your paper is missing here. 
  
## Feedback

For any queries or feedback, you can reach us at `csaba.toth@maths.ox.ac.uk` and `harald.oberhauser@maths.ox.ac.uk`.
  
