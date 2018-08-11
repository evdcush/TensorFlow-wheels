# TensorFlow Wheels

These are custom TensorFlow wheels built from source, optimized for my machines. All wheels were built for x64 linux (confirmed working on \*buntu 16.04) with intel processors.

## Wheels Available

You can find the wheels in the [releases page](https://github.com/evdcush/TensorFlow-wheels/releases).

* [**1.10.0, Python 3.6.\*, CUDA 9.1, cuDNN 7.1, MKL**](https://github.com/evdcush/TensorFlow-wheels/releases/tag/tf-1.10.0-gpu-mkl) Built for modern Intel architecture and Nvidia GPU
* [**1.8, Python 3.6.\*, CPU, MKL**](https://github.com/evdcush/TensorFlow-wheels/releases/tag/tf-1.8-cpu-ivybridge-MKL): Built for Ivy Bridge mobile processor (a ThinkPad T430), so it supports SSE4.1, SSE4.2, AVX
* [**1.8, Python 3.6.\*, CPU**](https://github.com/evdcush/TensorFlow-wheels/releases/tag/tf-1.8-cpu-westmere): Built for Westmere mobile processor (a ThinkPad x201), so it supports SSE4.1, SSE4.2


### Installing

Install wheels through pip:
`pip install --no-cache-dir tensorflow-<version>-cp36-cp36m-linux_x86_64.whl`

or

`pip install --no-cache-dir https://github.com/evdcush/TensorFlow-wheels/releases/download/<release>/<wheel>`


For [MKL](https://github.com/01org/mkl-dnn) installation, please reference [TinyMind's MKL install instructions](https://github.com/mind/wheels#mkl)
