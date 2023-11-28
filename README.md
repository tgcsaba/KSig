# <p align='center'> K<sub>Sig</sub> </p>
## <p align='center'> GPU-accelerated computation of popular time series kernels </p>
A [scikit-learn](https://github.com/scikit-learn/scikit-learn) compatible Python package, which provides a GPU-accelerated implementation for most powerful and popular time series kernels and features using [CuPy](https://github.com/cupy/cupy).

The time series kernels included in this package are:
- [Signature Kernel using dynamic programming](https://jmlr.org/papers/volume20/16-314/16-314.pdf), which computes truncated signature kernels exactly, see Algorithms 3 and 6;
- [Signature-PDE Kernel](https://arxiv.org/pdf/2006.14794.pdf), which approximates the untruncated signature kernel by solving a PDE;
- [Global Alignment Kernel](https://members.cbio.mines-paristech.fr/~jvert/publi/pdf/Cuturi2007Kernel.pdf), that computes a similarity score as a sum over all pairwise alignments between sequences, see equation 1.

Available time series features are:
- [Vanilla Path Signatures](https://arxiv.org/pdf/2206.14674.pdf) computing iterated integrals of a sequence lifted to a path by piecewise linear interpolation;
- [Low-Rank Signature Features](https://jmlr.org/papers/volume20/16-314/16-314.pdf) computes signature features using a low-rank algorithm which iteratively approximates outer products, see Algorithm 4;
- [Random Fourier Signature Features](https://arxiv.org/pdf/2311.12214.pdf) using Random Fourier Features and random projection approaches for tensors, see Algorithms 2 and 3;
- [Random Warping Series](https://proceedings.mlr.press/v84/wu18b/wu18b.pdf) that computes Dynamic Time Warping alignments between input sequences and random time series, see Algorithm 1.


## Introduction

The signature kernel is one of the most powerful similarity smeasure for sequences, which lifts a kernel on a given domain to **a kernel for sequences in that domain** with strong theoretical guarantees:
- It is a universal nonlinearity for time series, which means that it is flexible enough to approximate any continuous function on compact sets of sequences.
- Invariant to a natural transformation of time series called reparametrization (in the discrete setting often called time warping), but can be made sensitive to it by including time parametrization as an additional channel.

The signature kernel between sequences can be computed using an instantiation of the `ksig.kernels.SignatureKernel` (also, see `ksig.kernels.SignaturePDEKernel`) class by lifting a static kernel (i.e. in this case an RBF kernel for vector-valued data from the `ksig.static.kernels.RBFKernel` class) to a kernel for sequences as:
```python
import numpy as np
import ksig

# Number of signature levels to use.
n_levels = 5 

# Use the RBF kernel for vector-valued data as static (base) kernel.
static_kernel = ksig.static.kernels.RBFKernel() 

# Instantiate the signature kernel, which takes as input the static kernel.
sig_kernel = ksig.kernels.SignatureKernel(n_levels, static_kernel=static_kernel)

# Generate 10 sequences of length 50 with 5 channels.
n_seq, l_seq, n_feat = 10, 50, 5 
X = np.random.randn(n_seq, l_seq, n_feat)

# Sequence kernels take as input an array of sequences of ndim == 3,
# and work as a callable for computing the kernel matrix. 
K_XX = sig_kernel(X)  # K_XX has shape (10, 10).

# The diagonal kernel entries can also be computed.
K_X = sig_kernel(X, diag=True)  # K_X has shape (10,).

# Generate another array of 8 sequences of length 20 and 5 features.
n_seq2, l_seq2 = 8, 20
Y = np.random.randn(n_seq2, l_seq2, n_feat)

# Compute the kernel matrix between arrays X and Y.
K_XY = sig_kernel(X, Y)  # K_XY has shape (10, 8)

```

## Installation
We recommend setting up a fresh [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with Python 3.9/10 using
```
conda create -n ksig python=3.10 
conda activate ksig
```
In order to build `cupy` when installing the package, the `CUDA_PATH` environment variable must be properly set. The default CUDA installation path on Linux is `/usr/local/cuda-${CUDA-VERSION}/`, while on Windows it is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${CUDA_VERSION}\` where `${CUDA_VERSION}` is for example `12.3`. This can be done on Linux by (substitute for ${CUDA_VERSION}):
```
export CUDA_PATH=/usr/local/cuda-${CUDA-VERSION}/
```
Once this is done, the package can be installed by
```
pip install git+https://github.com/tgcsaba/ksig.git
```
Alternatively, if you are still unable to build `cupy`, you may be able to download a precompiled wheel for certain CUDA versions from [here](https://docs.cupy.dev/en/stable/install.html). Once this is done, you may try installing `ksig` package again using the previous line.

Finally, to make use of the functionality of the provided models under `ksig.models`, i.e. `PrecomputedKernelSVC` and `PrecomputedFeatureLinSVC`, you also need to install the `cuml` library. This can only be manually installed from [`RAPIDS`](https://docs.rapids.ai/install) repository. Here, scroll down to the section titled "Install RAPIDS" and select the `pip` install method, your CUDA version, your Python version, then select "Choose Specific Packages" and unselect every package besides `cuml`, and copy-paste the given command into the terminal where the `conda` environment is activated.
  
## More Examples

### Scaling to large datasets
A computational bottleneck associated with the full-rank signature kernel is a joint quadratic complexity in the number of training examples (N) and the length of the sequences (L), i.e. O(N<sup>2</sup>&middot;L<sup>2</sup>). Feature-based formulations of the signature kernel get around this issue by being able to represent the kernel using a finite-dimensional feature set.

One such example is [Random Fourier Signature Features](https://arxiv.org/pdf/2311.12214.pdf), which provides an unbiased approximation to the signature kernel, that converges in probability (1/M)-subexponentilly fast, where $M \in \mathbb{Z}_+$ is the truncation level.

This can be constructed using the `ksig.kernels.SignatureFeatures` class with two ingredients: 
- setting the `static_features` argument to an instance of the `ksig.static.features.RandomFourierFeatures` class,
- and the `projection` argument to an instance of `ksig.projections.TensorizedRandomProjection` (for RFSF-TRP) or to `ksig.projections.DiagonalProjection` (for RFSF-DP).
```python
import numpy as np
import ksig

# Number of signature levels to use.
n_levels = 5 

# Use 100 components in RFF and projection.
n_components = 100

# Instantiate RFF feature map.
static_feat = ksig.static.features.RandomFourierFeatures(n_components=n_components)
# Instantiate tensor random projections.
proj = ksig.projections.TensorizedRandomProjection(n_components=n_components)

# The RFSF-TRP feature map and kernel. Additionally to working as a callable for
# computing a kernel, it implements a fit and a transform method.
rfsf_trp_kernel = ksig.kernels.SignatureFetures(
    n_levels=n_levels, static_features=static_feat, projection=proj)

# Generate 1000 sequences of length 200 with 100 features.
n_seq, l_seq, n_feat = 1000, 200, 100
X = np.random.randn(n_seq, l_seq, n_feat)

# Fit the kernel to the data.
rfsf_trp_kernel.fit(X)

# Compute the kernel matrix as before.
K_XX = rfsf_trp_kernel(X)  # K_XX has shape (1000, 1000).

# GEnerate another array of 800 sequences of length 250 and 100 features.
n_seq2, l_seq2 = 800, 250
Y = np.random.randn(n_seq2, l_seq2, n_feat)

# Compute the kernel matrix between X and Y.
# The kernel does not have to be fitted a second time.
K_XY = rfsf_trp_kernel(X, Y)  # K_XY has shape (1000, 800)

# Alternatively, we may compute features separately for X and Y. Under the hood,
# this is what the call method does, i.e. compute features and take their inner product.
P_X = rfsf_trp_kernel.transform(X)  # P_X has shape (1000, 501)
P_Y = rfsf_trp_kernel.transform(Y)  # P_Y shape shape (800, 501)

# Check that the results match.
print(np.linalg.norm(K_XX - P_X @ P_X.T))
print(np.linalg.norm(K_XY - P_X @ P_Y.T))
```

## Results
The experiments can be run using the code provided in the `experiments` directory. The results displayed here are the ones appearing on [here](https://arxiv.org/pdf/2311.12214.pdf).

Results on [Multivariate UEA Datasets](https://timeseriesclassification.com/) with ($N \leq 1000$):
| | RFSF-DP | RFSF-TRP | KSig | KSigPDE | RWS | GAK | RBF | RFF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ArticularyWordRecognition | 0.984 | 0.981 | __0.990__ | 0.983 | _0.987_ | 0.977 | 0.977 | 0.978 |
| AtrialFibrillation | 0.373 | 0.320 | _0.400_ | 0.333 | __0.427__ | 0.333 | 0.267 | 0.373 |
| BasicMotions | __1.000__ | __1.000__ | __1.000__ | __1.000__ | _0.995_ | __1.000__ | 0.975 | 0.860 |
| Cricket | 0.964 | 0.964 | 0.958 | _0.972_ | __0.978__ | 0.944 | 0.917 | 0.886 |
| DuckDuckGeese | 0.636 | _0.664_ | __0.700__ | 0.480 | 0.492 | 0.500 | 0.420 | 0.372 |
| ERing | 0.921 | 0.936 | 0.841 | _0.941_ | __0.945__ | 0.926 | 0.937 | 0.915 |
| EigenWorms | _0.817_ | __0.837__ | 0.809 | 0.794 | 0.623 | 0.511 | 0.496 | 0.443 |
| Epilepsy | _0.949_ | 0.942 | __0.949__ | 0.891 | 0.925 | 0.870 | 0.891 | 0.777 |
| EthanolConcentration | 0.457 | 0.439 | __0.479__ | _0.460_ | 0.284 | 0.361 | 0.346 | 0.325 |
| FingerMovements | 0.608 | 0.624 | __0.640__ | _0.630_ | 0.612 | 0.500 | 0.620 | 0.570 |
| HandMovementDirection | _0.573_ | 0.568 | __0.595__ | 0.527 | 0.403 | __0.595__ | 0.541 | 0.454 |
| Handwriting | 0.434 | 0.400 | 0.479 | 0.409 | __0.591__ | _0.481_ | 0.307 | 0.249 |
| Heartbeat | 0.717 | 0.712 | 0.712 | __0.722__ | 0.714 | 0.717 | 0.717 | _0.721_ |
| JapaneseVowels | 0.978 | 0.978 | __0.986__ | __0.986__ | 0.955 | _0.981_ | _0.981_ | 0.979 |
| Libras | 0.898 | __0.928__ | _0.922_ | 0.894 | 0.837 | 0.767 | 0.800 | 0.800 |
| MotorImagery | _0.516_ | __0.526__ | 0.500 | 0.500 | 0.508 | 0.470 | 0.500 | 0.482 |
| NATOPS | 0.906 | 0.908 | 0.922 | __0.928__ | _0.924_ | 0.922 | 0.917 | 0.900 |
| PEMS-SF | 0.800 | 0.808 | 0.827 | _0.838_ | 0.701 | __0.855__ | __0.855__ | 0.770 |
| RacketSports | 0.874 | 0.861 | __0.921__ | _0.908_ | 0.878 | 0.849 | 0.809 | 0.755 |
| SelfRegulationSCP1 | 0.868 | 0.856 | _0.904_ | _0.904_ | 0.829 | __0.915__ | 0.898 | 0.885 |
| SelfRegulationSCP2 | 0.489 | 0.510 | _0.539_ | __0.544__ | 0.481 | 0.511 | 0.439 | 0.492 |
| StandWalkJump | 0.387 | 0.333 | _0.400_ | _0.400_ | 0.347 | 0.267 | __0.533__ | 0.267 |
| UWaveGestureLibrary | 0.882 | 0.881 | __0.912__ | 0.866 | _0.897_ | 0.887 | 0.766 | 0.846 |
| Avg.acc. | _0.740_ | 0.738 | __0.756__ | 0.735 | 0.710 | 0.702 | 0.692 | 0.656 |
| Avg.rank | 3.652 | 3.739 | __2.348__ | _2.957_ | 4.043 | 4.217 | 4.913 | 5.957 |

Results on [Multivariate UEA Datasets](https://timeseriesclassification.com/) with ($N > 1000$), [`fNIRMS2MW`](https://github.com/tufts-ml/fNIRS-mental-workload-classifiers) ($N = 10^5$), [`SITS11M`](https://cloudstor.aarnet.edu.au/plus/index.php/s/pRLVtQyNhxDdCoM) ($N = 10^6$):

| | RFSF-DP | RFSF-TRP | RWS | RFF |
| --- | --- | --- | --- | --- |
| CharacterTrajectories | _0.990_ | _0.990_ | __0.991__ | 0.989 |
| FaceDetection | _0.653_ | __0.656__ | 0.642 | 0.572 |
| InsectWingbeat | _0.436_ | __0.459__ | 0.227 | 0.341 |
| LSST | 0.589 | _0.624_ | __0.631__ | 0.423 |
| PenDigits | _0.983_ | 0.982 | __0.989__ | 0.980 |
| PhonemeSpectra | _0.204_ | _0.204_ | __0.205__ | 0.083 |
| SITS1M | __0.745__ | _0.740_ | 0.610 | 0.718 |
| SpokenArabicDigits | __0.981__ | _0.980_ | __0.981__ | 0.964 |
| fNIRS2MW | __0.659__ | _0.658_ | 0.621 | 0.642 |
| Avg.acc. | _0.693_ | __0.699__ | 0.655 | 0.635 |
| Avg.rank | __1.778__ | _1.889_ | 2.222 | 3.333 |

## Documentation
Coming very soon...

## Literature

A non-exhaustive list of some papers that involve signature kernels in one way or another (in chronological order):

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

- [The Signature Kernel](https://arxiv.org/pdf/2305.04625.pdf) by Lee, Oberhauser

- [Non-adversarial training of Neural SDEs with
signature kernel scores](https://arxiv.org/pdf/2305.16274.pdf) by Issa, Horvath, Lemercier, Salvi

- [Neural signature kernels as infinite-width-depth-limits of controlled ResNets](https://arxiv.org/pdf/2303.17671.pdf) by Cirone, Lemercier, Salvi

- [Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures](https://arxiv.org/pdf/2306.15835.pdf) by Horvath, Issa

- [Random Fourier Signature Features](https://arxiv.org/pdf/2311.12214.pdf) by Toth, Oberhauser, Szabo

Contact us if you woud like your paper or project to appear here. 
  
## Contacts

For queries or feedback, you can reach us at `csaba.toth@maths.ox.ac.uk` and `harald.oberhauser@maths.ox.ac.uk`.
  
