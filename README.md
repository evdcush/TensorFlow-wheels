# TensorFlow Wheels

These are custom TensorFlow wheels built from source, optimized for my machines. All wheels were built for x64 linux (confirmed working on \*buntu 16.04) with intel processors.

The GPU versions are built for a machine with a Core i7-7700 and GTX 1070.

The CPU versions are mostly built for my ancient thinkpads (T430: -march=ivybridge, X201: -march=westmere).

## Wheels Available

You can find the wheels in the [releases page](https://github.com/evdcush/TensorFlow-wheels/releases).

* * *

## GPU builds
| Version |  Py | CUDA | cuDNN |      TensorRT      |         MKL        | Link                                                                                                                                        |
|:-------:|:---:|:----:|:-----:|:------------------:|:------------------:|---------------------------------------------------------------------------------------------------------------------------------------------|
|  1.10.0 | 3.6 |  9.2 |  7.2  | :heavy_check_mark: | :heavy_check_mark: | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.10.0-gpu-9.2-tensorrt-mkl/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl |
|  1.10.0 | 3.6 |  9.1 |  7.1  |         :x:        | :heavy_check_mark: | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.10.0-gpu-9.1-mkl/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl          |



## CPU builds
| Version |  Py |       SSE4.1       |       SSE4.2       |         AVX        | AVX2 | FMA |         MKL        | Links                                                                                                                                    |
|:-------:|:---:|:------------------:|:------------------:|:------------------:|:----:|:---:|:------------------:|------------------------------------------------------------------------------------------------------------------------------------------|
|  1.10.0 | 3.6 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :x: | :x: | :heavy_check_mark: | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.10.0-cpu-mkl-ivybridge/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl |
|   1.8   | 3.6 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :x: | :x: | :heavy_check_mark: | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.8-cpu-ivybridge-MKL/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl     |
|   1.8   | 3.6 | :heavy_check_mark: | :heavy_check_mark: |         :x:        |  :x: | :x: |         :x:        | https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.8-cpu-westmere/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl          |



## Installing TensorFlow wheels:

### Install wheel via pip
#### From the directory of the downloaded wheel:
`pip install --no-cache-dir tensorflow-<version>-cp36-cp36m-linux_x86_64.whl`

#### From the direct link to the wheel:
`pip install --no-cache-dir https://github.com/evdcush/TensorFlow-wheels/releases/download/<release>/tensorflow-<version>-cp36-cp36m-linux_x86_64.whl`

* * *



# Notes on Building TF, and setting up your environment
Quite a few people have asked me how I build TF, and I myself found the resources online to be either incomplete or even incorrect when I was first learning. So I took some notes on the process.

The following guide is the quick and dirty rundown on building tf, and setting up your environment. For more detailed steps, checkout the [extended setup guide](env-setup-guide.md)


## Preparing your environment
- **Install GCC-7** (GCC-8 is not supported by TF, and anything less than GCC-7 will not support `-march=skylake`)
- **Setup your python environment** (I highly suggest [pyenv](https://github.com/pyenv/pyenv))
  - Make sure you have `keras` installed, else your build will fail at the very end and you will be heartbroken
  - Also make sure you don't have any other existing TF installations (like `tf-nightly` that can be installed as a dependency for other packages like `tangent`)
- **For GPU**:
  - Setup CUDA (9.2)
  - Setup cuDNN (7.2)
  - (optionally setup TensorRT)
- (optionally install MKL)
- **Install Bazel** (I chose the `apt` method over source)

### Tensorflow configuration and build: You've finished setup; now it's time for building TF.
First, clone the tensorflow source
```bash
#==== Clone tensorflow source
git clone --depth=1 https://github.com/tensorflow/tensorflow.git && cd tensorflow
```
Now, build tensorflow based on your needs

## Simple build: no need for ./configure, (the majority of cases)
Go this route if the following sounds true:
- I don't need XLA JIT, GDR, VERBS, OpenCL SYSCL, or MPI support
- I don't need GPU (CUDA) support
  - Or, I need GPU support, and I have CUDA 9.0 and don't need TensorRT

### Build cases:
- > I don't need GPU support, nor XLA JIT, GDR, VERBS, OpenCL SYSCL, MPI support
  - `bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

- > But I wanted MKL support!
  - `bazel build --config=opt --config=mkl //tensorflow/tools/pip_package:build_pip_package`

- > I'm actually building for my old-ass, 1st-gen Core i\*, thinkpad
  - `bazel build -c opt --copt=-march="westmere" //tensorflow/tools/pip_package:build_pip_package`
    - [*complete list of GCC march options*](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#x86-Options)

- > I have vanilla CUDA 9, no TensorRt, oh and let's do MKL just for kicks
  - `bazel build --config=opt --config=cuda --config=mkl //tensorflow/tools/pip_package:build_pip_package`
    - I say "just for kicks" because [mkl does nt work when also using `--config=cuda`](https://github.com/tensorflow/docs/blob/master/site/en/performance/performance_guide.md#optimizing-for-cpu). I always build my GPU wheels with MKL anyway, :sunglasses: :ok_hand:



## Specialized build: ./configure
You'll probably know if you need to go this route. But do so if the following sounds true:
- I need XLA JIT, GDR, VERBS, OpenCL SYSCL, or MPI support
- I need GPU support for my CUDA 9.2, TensorRT setup
- I have a non-geforce, Nvidia GPU (or any nvidia GPU with a compute capability different from 6.1)

`./configure` is actually pretty straightforward. Just answer y/n for the stuff you want, and provide any additional info it may ask for your use-case.
```bash
# eg: I want XLA JIT
Do you wish to build TensorFlow with XLA JIT support? [y/N]: y

# I want nGraph
Do you wish to build TensorFlow with nGraph support? [y/N]: y
```

#### CUDA options:
```
# I'm on CUDA 9.2
Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 9.2

# Default for paths (DO NOT PUT /usr/local/cuda-9-2 !)
Please specify the location where CUDA 9.2 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.2

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

# TensorRT
Do you wish to build TensorFlow with TensorRT support? [y/N]: y

Please specify the location where TensorRT is installed. [Default is /usr/lib/x86_64-linux-gnu]:

# NCCL: I use 1.3, so that's what I specify
Please specify the NCCL version you want to use. If NCCL 2.2 is not installed, then you can use version 1.3 that can be fetched automatically but it may have worse performance with multiple GPUs. [Default is 2.2]: 1.3

# CUDA compute capability: just do whatever your card is rated for, mine's 6.1
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 6.1]:
```


### Hey... What about that one line though? The optimization flags?
```bash
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
```
**Don't put anything here**, just smash that enter key.
Anything you would pass here is either automatically handled by your answers in ./configure, or specified to the bazel build args

### Once you've finished ./configuration, just call bazel build
`bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

If you built for CUDA or whatever, it's already been setup. There's no need to specify it as a config option again, nor is there any need to list out all the ISA extensions (SSE4.2, AVX, etc.). That's all handled by -march=native (that line you were SUPPOSED to default). If you want mkl, you can `bazel build --config=opt --config=mkl`

## Finally: build the pip wheel
I like just putting the pip whl in home, but put it wherever you want.

`bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/`

* * *

### The extended guide to setup and building
If you've been having trouble with setup or building TF, or you're new to this sort of thing, or you just want to see my method, then checkout my [extended setup guide](env-setup-guide.md) which has detailed steps on the process, from GCC to Python env to CUDA and MKL.


## License:
Except where noted, the written content of this repo, is licensed as [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/), while any software/code is licensed as BSD-3.
