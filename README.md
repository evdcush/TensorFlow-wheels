# TensorFlow Wheels

These are custom TensorFlow wheels built from source, optimized for my machines. All wheels were built for x64 linux with intel processors.

The GPU versions are built for a machine with a Core i7-7700 and GTX 1070.

The CPU versions are mostly built for my ancient thinkpads (T430: `-march=ivybridge`, X201: `-march=westmere`).

## Wheels Available

You can find the wheels in the [releases page](https://github.com/evdcush/TensorFlow-wheels/releases).

* * *

## GPU builds
| Version | buntu |  Py | CUDA | cuDNN |      TensorRT      | AdditionalOpts      | Link                                                                                                                                        |
|---------|:-----:|:---:|:----:|:-----:|:------------------:|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 1.13.1  | 18.04 | 3.7 | 10.0 |  7.5  | :heavy_check_mark: | - XLA JIT<br/>- MKL | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.13.1-py37-gpu-10.0/tensorflow-1.13.1-cp37-cp37m-linux_x86_64.whl        |
| 1.12.0  | 18.04 | 3.7 | 10.0 |  7.3  | :heavy_check_mark: | - XLA JIT<br/>- MKL | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.12.0-py37-gpu-10.0/tensorflow-1.12.0-cp37-cp37m-linux_x86_64.whl        |
| 1.12.0  | 18.04 | 3.6 | 10.0 |  7.3  | :heavy_check_mark: | - XLA JIT<br/>- MKL | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.12.0-gpu-10.0/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl             |
| 1.11.0  | 18.04 | 3.6 | 10.0 |  7.3  | :heavy_check_mark: | - MKL               | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.11.0-gpu-10.0_7.3_5.0-mkl/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl |
| 1.10.0  | 16.04 | 3.6 |  9.2 |  7.2  | :heavy_check_mark: | - XLA JIT<br/>- MKL | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.10.0-gpu-9.2-tensorrt-mkl/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl |
| 1.10.0  | 16.04 | 3.6 |  9.1 |  7.1  |         :x:        | - MKL               | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.10.0-gpu-9.1-mkl/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl          |



## CPU builds
| Version | buntu |  Py |       SSE4.1       |       SSE4.2       |         AVX        | AVX2 | FMA |         MKL        | Links                                                                                                                                     |
|:-------:|:-----:|:---:|:------------------:|:------------------:|:------------------:|:----:|:---:|:------------------:|-------------------------------------------------------------------------------------------------------------------------------------------|
|  1.13.1 | 18.04 | 3.7 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :x: | :x: | :heavy_check_mark: | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.13.1-py37-cpu-ivybridge/tensorflow-1.13.1-cp37-cp37m-linux_x86_64.whl |
|  1.13.1 | 18.04 | 3.7 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :x: | :x: | :heavy_check_mark: | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.13.1-py37-cpu-westmere/tensorflow-1.13.1-cp37-cp37m-linux_x86_64.whl  |
|  1.12.0 | 18.04 | 3.7 | :heavy_check_mark: | :heavy_check_mark: |         :x:        |  :x: | :x: |         :x:        | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.12.0-py37-cpu-westmere/tensorflow-1.12.0-cp37-cp37m-linux_x86_64.whl  |
|  1.12.0 | 16.04 | 3.7 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :x: | :x: | :heavy_check_mark: | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.12.0-py37-cpu-ivybridge/tensorflow-1.12.0-cp37-cp37m-linux_x86_64.whl |
|  1.10.0 | 16.04 | 3.6 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :x: | :x: | :heavy_check_mark: | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.10.0-cpu-mkl-ivybridge/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl  |
|   1.8   | 16.04 | 3.6 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :x: | :x: | :heavy_check_mark: | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.8-cpu-ivybridge-MKL/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl      |
|   1.8   | 16.04 | 3.6 | :heavy_check_mark: | :heavy_check_mark: |         :x:        |  :x: | :x: |         :x:        | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.8-cpu-westmere/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl           |


# Installing TensorFlow wheels:

## Install wheel via pip
### From the directory of the downloaded wheel:

`pip install --no-cache-dir tensorflow-<version>-<py-version>-linux_x86_64.whl`

### From the direct link to the wheel:

`pip install --no-cache-dir https://github.com/evdcush/TensorFlow-wheels/releases/download/<release>/tensorflow-<version>-<py-version>-linux_x86_64.whl`

* * *

* * *


# Notes on Building TF from Source: making your own wheel :wrench:
Quite a few people have asked me how I build TF, and I myself found the resources online to be either incomplete or even incorrect when I was first learning. So I took some notes on my own process. I've managed to reach some degree of consistency when building in the manner described, but YMMV.


The following guide is the quick and dirty rundown on building tf, and setting up your environment. For more detailed steps, checkout the [extended setup guide](env-setup-guide.md).


## Preparing your environment
- **Install GCC-7** (GCC-8 is not supported by TF, and anything less than GCC-7 will not support `-march=skylake`)
- **Setup your python environment** (I highly suggest [pyenv](https://github.com/pyenv/pyenv))
  - Make sure you have `keras` installed, else your build will fail at the very end and you will be heartbroken
  - Also make sure you don't have any other existing TF installations (like `tf-nightly` that can be installed as a dependency for other packages)
- **For GPU**:
  - Setup CUDA
  - Setup cuDNN
  - Setup TensorRT
- (optionally install MKL)
- **Install Bazel** <= `0.21`, (`0.22` not supported by TF)

* * *

# Tensorflow configuration and build
You've finished setup; now it's time for building TF.

### First, clone the tensorflow source.
The tensorflow repo is large (approx 300mb) and can take awhile to download the full git tree, so I *highly* recommend cloning your target version tag at `--depth=1`.

Regardless how how you clone TF, you must **checkout the version you wish to build**. So not only is it faster/less-space to clone your target version, you don't need to check out the right branch--that version is the only branch you have.


```bash
# RECOMMENDED: clone target version only, eg v1.13.1
git clone https://github.com/tensorflow/tensorflow.git --branch v1.13.1 --depth=1

# Full repo clone & checkout process
git clone https://github.com/tensorflow/tensorflow.git && cd tensorflow
git checkout v1.13.1

```
Now, build tensorflow based on your needs


## Simple build: no need for ./configure
As of TensorFlow 1.13.1, **the majority of custom build options** have been abstracted from the `./configure` process to convenient preconfigured Bazel build configs. Furthermore, GPU versions are now built against the latest CUDA libraries (CUDA 10, cuDNN 7, TensorRT 5.0). :tada: :ok_hand:


This means that unless you need support for the following:
- OpenCL SYCL
- ROCm
- MPI

**You don't need to go through configure**

## Build examples:
:warning: **NB**: When building for other machines, keep in mind that _**your distro versions generally need to match**_. ie, building for 14.04 from 16.04 will likely not work.

This is because your `GLIBC` versions will not be matched. I've been trying for awhile to configure bazel for compiling against a linked GLIBC (vs system lib) or to compile for a target GLIBC version, but have not been successful yet.

### *"I don't need any extra stuff or GPU support, just a TF optimized to my machine"*

:point_right: `bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

* * *

### *"But I wanted MKL support!"*

:point_right: `bazel build --config=opt --config=mkl //tensorflow/tools/pip_package:build_pip_package`

* * *

### *"I'm actually building for my old-ass, 1st-gen Core i\*, thinkpad"*
:point_right: `bazel build --copt=-march="westmere" -c opt //tensorflow/tools/pip_package:build_pip_package`
  - [*complete list of GCC march options*](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#x86-Options)
  - **NB**: GLIBC for target platform must match build platform. Unless you are a linux wizard or bazel god, you probably won't be able to build for older GLIBC.

* * *

### *"I have the default CUDA library versions for this TF version. Oh and let's do MKL just for kicks"*

:point_right: `bazel build --config=opt --config=cuda --config=mkl //tensorflow/tools/pip_package:build_pip_package`
  - I say "just for kicks" because [mkl does nt work when also using `--config=cuda`](https://github.com/tensorflow/docs/blob/master/site/en/performance/performance_guide.md#optimizing-for-cpu). I always build my GPU wheels with MKL anyway, :sunglasses: :ok_hand:


* * *

## Specialized build: ./configure
Configure your Bazel build through `configure` if the following is true:
- I need OpenCL SYSCL, ROCm, or MPI support
- My GPU library versions are different from TF defaults: TF 1.13.1 (`CUDA 10.0, cuDNN 7.3, TensorRT 5.0, nccl 2.2`)
  - This is usually the only reason I go through `configure`

`./configure` is actually pretty straightforward. Just answer y/n for the stuff you want, and provide any additional info it may ask for your use-case.
```bash
# eg: I want Open SYCL
Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: y
OpenCL SYCL support will be enabled for TensorFlow.

# I want ROCm
Do you wish to build TensorFlow with ROCm support? [y/N]: y
ROCm support will be enabled for TensorFlow.

```

## CUDA options:
Use default paths. Whatever version of CUDA you have installed, it always links to the default `/usr/local/cuda` path. If you did not use default pathing during your CUDA setup, you probably already know what you are doing.

All lines with `#` are my own comments
```
# I'm on CUDA 10.0
Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 10.0

# Default for paths (DO NOT PUT /usr/local/cuda-10-0 !)
Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 7.3

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

Do you wish to build TensorFlow with TensorRT support? [y/N]: y

Please specify the location where TensorRT is installed. [Default is /usr/lib/x86_64-linux-gnu]:

Please specify the NCCL version you want to use. If NCCL 2.2 is not installed, then you can use version 1.3 that can be fetched automatically but it may have worse performance with multiple GPUs. [Default is 2.2]: 2.3

NCCL libraries found in /usr/lib/x86_64-linux-gnu/libnccl.so
This looks like a system path.
Assuming NCCL header path is /usr/include

# CUDA compute capability: just do whatever your card is rated for, mine's 6.1
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 6.1]:

```


## Hey... What about that one line though? The optimization flags?
```bash
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
```


**Don't put anything here**, just smash that enter key :hammer:

Anything you would pass here is either automatically handled by your answers in ./configure, or specified to the bazel build args.


### Once you've finished ./configuration, just call bazel build

`bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

If you built for CUDA or whatever, it's already been setup. There's no need to specify it as a config option again, nor is there any need to list out all the ISA extensions (SSE4.2, AVX, etc.). That's all handled by -march=native (that line you were SUPPOSED to default). If you want mkl, you can `bazel build --config=opt --config=mkl`

### Non-native `march`
If you want to build for a different target `march`, and you smashed that `"--config=opt"` line, this is where you specify the desired `march`, eg:

`bazel build --copt=-march="sandybridge" //tensorflow/tools/pip_package:build_pip_package`

## Finally: build the pip wheel
I like just putting the pip whl in home, but put it wherever you want.

`bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/`

* * *

### The extended guide to setup and building
If you've been having trouble with setup or building TF, or you're new to this sort of thing, or you just want to see my method, then checkout my [extended setup guide](env-setup-guide.md) which has detailed steps on the process, from GCC to Python env to CUDA and MKL.


## License:
Except where noted, the written content of this repo, is licensed as [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/), while any software/code is licensed as BSD-3.
