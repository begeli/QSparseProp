# QSparseProp: Quantized Backpropagation for Sparse Neural Networks

This repo contains the code for my master's thesis 
where I implemented a quantized backpropagation algorithm
to speed up finetuning of sparse neural networks. We support 
quantization for sparse linear and convolutional layers.

Neural networks are growing at exponential rates. It is becoming more resource 
intensive and time consuming to train and deploy more sophisticated models. In order
to alleviate this issue, many model compression methods have been developed. These
methods make model development both faster and cheaper. Two such methods are
quantization and pruning. Unfortunately, most quantization and pruning methods 
remain theoretical without system support. This thesis introduces a high-performance
library with quantized and sparse kernels to speed up neural network finetuning. Our
library provides efficient versions of the backpropagation algorithm for linear and 
convolutional modules. Our algorithms apply to unstructured sparsity and implement
8-bit integer quantization kernels for forward and backward passes. Models trained 
using our framework provide up to 2x-4x speed ups while halving the memory allocation
with acceptable accuracy loss.

## Results

- Decreased finetuning time by 50%.
- Decreased memory usage by 50%.
- Maintained 96.7% of the baseline top-1 accuracy. (Datasets used: CIFAR10, CIFAR100, MNIST, CALTECH101, Plant Diseases)

### Experiment Setup

We conducted our experiments using an Intel® Core™ i7-
1068NG7 processor, which uses the Icelake architecture. It has a 
base frequency of 2.3 GHz. The cache sizes for L1, L2 and L3 
caches are 80KiB (32 KiB I + 48 KiB D), 512KiB 
and 8MiB (shared), respectively. L1 and L2 caches are 8-way 
associative, and L3 cache is 16-way associative. The maximum memory bandwidth is 58.3GB/s. 
We turn off the Intel Turbo Boost technology throughout the timing performance experiments. 

- OpenMP version: 16.0.5
- C++ version: C++17
- C++ compiler version: clang version 14.0.3
- Python version: 3.9.17
- PyBind11 version: 2.10.4

## Building C++ Backend

To build the C++ code we use cmake version 3.24.2. 

To build the C++ backend in debug mode we use the following command:
``` 
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=<PATH-TO-BUILD-TOOL (i.e. /Applications/CLion.app/Contents/bin/ninja/mac/ninja)> -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DTESTING=1 -DSTOCHASTIC_ROUNDING_DISABLED=1 "-DCMAKE_CXX_FLAGS=-std=c++17 -O3 -march=<Machine Architecture (i.e. icelake-client)> -mprefer-vector-width=512 -ffast-math -fno-finite-math-only -Xclang -fopenmp" -G <Build System Generator (We used Ninja)> -S <PATH-TO-SRC>/qspraseprop_backend -B <PATH-TO-BUILD-FOLDER>
```
Debug mode allows us to run the unit tests.

To build the C++ backend in release mode we use the following command:
``` 
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=<PATH-TO-BUILD-TOOL> -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ "-DCMAKE_CXX_FLAGS=-std=c++17 -march=<Machine Architecture> -mprefer-vector-width=512 -O3 -ffast-math -fno-finite-math-only -Xclang -fopenmp" -G <Build System Generator (We used Ninja)> -S <PATH-TO-SRC>/qspraseprop_backend -B <PATH-TO-BUILD-FOLDER>
``` 

Note: -Xclang flag is only needed for MacOS.
Note: The user should change the paths in ```openmp.cmake``` 
file in the ```cmake``` folder. Make sure the paths in ```include_directories```
and ```find_library``` commands point to the OpenMP library in your system.

## Generating the Python Module from the QSparseProp backend

This step requires a Python Interpreter and Setuptools package.
Python version is 3.9.17 as stated above. Setuptools version is 65.5.1. 
If it is not already installed, you can install it with
```
pip install setuptools==65.5.1
```

To create the Python package change directory to "qsparseprop_backend." 
Python wheel which contains the QSparseProp implementation can be created using the command:
```
python bdist_wheel setup.py
```

"setup.py" contains the build flags required to compile the C++ library.
Make sure the paths to cmake, C++ compiler, compiler flags and build tool are correct. 
In other words, make sure these parameters match paths and flags in the "Building C++ Backend" commands.

After running the command there should be a "/dist" folder created inside the qsparseprop_backend folder.
Inside this folder, you can find the Python package with ".whl" extension.

## Usage of Python Module

Python package can be installed into a Python project using the 
following command:

``` 
pip install <PATH-TO-WHEEL-FILE>/<NAME-OF-THE-WHEEL-FILE>
```

**Sparsifying a single linear layer**

If you have a _Linear_ module called ```linear```, you can convert it into 
_Quantized Sparse Linear_ module using the ```from_dense``` method.

```
from modules.linear import QuantizedSparseLinear

qsparse_linear = QuantizedSparseLinear.from_dense(linear)
```

Parameters of the ```linear``` module will be automatically stored in the 
new module using the _CSR_ format. 

**Sparsifying a single convolutional layer**

Convolutional layer sparsification is similar to Linear layer sparsification. 
The only difference is, ```from_dense``` function for convolutional layers 
has a parameter called ```vectorizing_over_on``` which is used to determine
the iteration order of the dimensions of tensors during forward and backward
passes. If this parameter is set to _True_, the input tensor (and its gradient tensor)
will be permuted such that the batch dimension and the innermost dimension
are swapped. This helps with vectorizing the convolution operations when 
the innermost dimension is not large enough to fit into a vector. 

Parameters are stored in a ```CSR-like``` sparse format.

```
from modules.conv2d import QuantizedSparseConv2d

qsparse_linear = QuantizedSparseConv2d.from_dense(conv, vectorizing_over_on=False)
```

**Sparsifying the entire network**

We replace each _Linear_ or _Conv2d_ layer in a network with a sparse one, if the following conditions are met:

1) It is at least 80% sparse.
2) The sparse module is faster than the original dense one (in terms of forward+backward time).

This behavior is implemented in the swap_modules_with_sparse method 
in utils.network_utils. For example, if you have a sparse (global or uniform) model:

```
from utils.network_utils import swap_modules_with_sparse

model = swap_modules_with_sparse(model, input_shape, inplace=True, verbose=True, prototype=False, quantize=True)
```

Notice that you need to provide the input_shape to this method, 
which is easily accessible through your DataLoader. 
The swap_modules_with_sparse method will iterate through the 
network's layers and replace them with their sparse counterparts 
if the above two conditions are met.
