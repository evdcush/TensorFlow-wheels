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

* * *

# Installing TensorFlow wheels:
### Install wheel via pip
#### From the directory of the downloaded wheel:
`pip install --no-cache-dir tensorflow-<version>-cp36-cp36m-linux_x86_64.whl`

#### From the direct link to the wheel:
`pip install --no-cache-dir https://github.com/evdcush/TensorFlow-wheels/releases/download/<release>/tensorflow-<version>-cp36-cp36m-linux_x86_64.whl`


For [MKL](https://github.com/01org/mkl-dnn) installation, please reference [TinyMind's MKL install instructions](https://github.com/mind/wheels#mkl)

* * *

# TODO/WIP

# Notes on Building TF, and setting up your environment
## Building TensorFlow from source
Why? You probably got those warnings about tensorflow not being optimized for certain instruction sets from tf when you initialized your tf session. Or maybe you wanted support for other services, platforms, or libraries. Maybe you just want a bespoke tensorflow for your exact setup.

The build process is not trivial, but if you've managed to setup all the dependencies and have gone through the build process a few times, it is fairly straightfoward.


**Preparing your environment**: If you haven't setup your environment yet, you can take a look at my notes on that below. Here's the quick rundown:
- Install GCC-7 (GCC-8 is not supported by TF, and anything less than GCC-7 will not support `-march=skylake`)
- Setup python environment (I suggest pyenv)
  - Make sure you have `keras` installed, else your build will fail at the very end
- For GPU:
  - Setup CUDA (9.2)
  - Setup cuDNN (7.2)
  - (optionally setup TensorRT)
- Install Bazel (I chose apt over source)
- Clone the tensorflow repo: `git clone --depth=1 https://github.com/tensorflow/tensorflow.git`

You are now ready to smash that configure.

### Understanding optimization flags
gcc -march=<my_arch> -Q --help=target

So, you've jumped through all the hoops and now you're in `./configure`, and you've reached the line:
> Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:





# Environment setup and library installation notes
## GCC setup
It's not strictly necessary to update GCC. But if your CPU is newer than Broadwell, your GCC -march=native is still configured for broadwell. Only `gcc-7` and above support newer architectures. You should be on `gcc-8`. But when you are building tensorflow, make sure you switch to `gcc-7`, as `gcc-8` is not yet a supported compiler for tensorflow. Follow these instructions to install both GCC versions, and to be able to switch between them.
```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt full-upgrade
sudo apt install gcc-7 g++-7 gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100 --slave /usr/bin/g++ g++ /usr/bin/g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
# To switch between gcc-7 and gcc-8:
sudo update-alternatives --config gcc
```

## Setting up your python environment
```bash
#==== First, dependencies
sudo apt install -y curl apt-transport-https cmake make git-core autoconf automake build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev xz-utils tk-dev

#==== Installing Pyenv
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash;
# Add the lines suggested by the installer to your shell config file (eg, uncommented of course:)
# export PATH="/home/my_user_name/.pyenv/bin:$PATH"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"
source ~/.zshrc # or .bashrc or whatever
pyenv install 3.6.6 # tensorflow does not currently support 3.7
pyenv virtualenv 3.6.6 my_virtualenv_name
pyenv local my_virtualenv_name
pip install -U pip setuptools
pip install wheel numpy scipy keras # MUST install wheel and keras for tf to build successfully
```


## Installing CUDA, cuDNN, TensorRT
### Cuda
I am assuming a clean build here (from a fresh install). If you have a preexisting CUDA installation, or nvidia drivers, the process is more complicated.
```bash
#==== Install CUDA deb
# Make sure you download the network package, NOT the local
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt update

# WARNING: if you install the normal `cuda` package instead of the
#  specific version package, the package manager will automatically
#  update your CUDA installation. Very rarely do CUDA-accelerated
#  libraries support the latest CUDA versions, so you want to lock
#  in your cuda version
sudo apt install cuda-9-2

# Now, add the following lines to your shell config (.zshrc, etc)
#export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
reboot

# Verify successful installation
nvcc -V
nvidia-smi
```

### cuDNN
cuDNN installation always comes after you've installed CUDA. cuDNN can be a bit trickier than the CUDA installation, I had trouble with this at first, but I think the trick here is to use the tar files instead of the other formats (.deb, .local, or whatever).

You also have to sign up and do a silly survey to get access to the cuDNN (and TensorRT) downloads. After you've done that, grab the cuDNN tar file supporting your CUDA version.
```bash
#==== cuDNN
# tar file method:
tar -xzvf cudnn-9.2-linux-x64-v7.2.1.38.tgz # whatever ver you got
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# Now verify cuDNN installation was successful
#  For cuDNN 7.2, you should see CUDNN_MAJOR 7, CUDNN_MINOR 2
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

## TensorRT
While CUDA and cuDNN are required for any GPU-accelerated ML framework, TensorRT is an optional library. But if you've come this far, why not squeeze a little more from your GPU? Plus, it's the easiest installation out of the three.
```bash
#==== TensorRT
# First install CUDA. Then get the latest TensorRT supporting your
#  CUDA version. eg, at the time of writing:
sudo dpkg -i nv-tensorrt-repo-ubuntu1604-cuda9.2-ga-trt4.0.1.6-20180612_1-1_amd64.deb
sudo apt update
sudo apt install tensorrt
sudo apt install uff-converter-tf
# verify:
dpkg -l | grep TensorRT
```

All done with the GPU environment!


## Installing MKL
