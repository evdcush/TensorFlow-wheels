# TensorFlow Wheels

These are custom TensorFlow wheels built from source, optimized for my machines. All wheels were built for x64 linux (confirmed working on \*buntu 16.04) with intel processors.

## Wheels Available

You can find the wheels in the [releases page](https://github.com/evdcush/TensorFlow-wheels/releases).

* [**1.10.0, Python 3.6.\*, CUDA 9.1, cuDNN 7.1, MKL**](tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl): The wheel in the project root directory. GTX 1070 (cuda compute 6.1), Kaby Lake i7.
* [**1.8, Python 3.6.\*, CPU, MKL**](older_architecture/Ivy\ Bridge/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl): Built for Ivy Bridge mobile processor (in my ThinkPad T430), so it supports SSE4.1, SSE4.2, AVX
* [**1.8, Python 3.6.\*, CPU**](older_architecture/Westmere/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl): Built for Westmere mobile processor (in my ThinkPad x201), so it supports SSE4.1, SSE4.2


### Installing

Install wheels through pip:
`pip install --no-cache-dir tensorflow-<version>-cp36-cp36m-linux_x86_64.whl`

or

`pip install --no-cache-dir https://github.com/evdcush/TensorFlow-wheels/releases/download/<release>/<wheel>`


For [MKL](https://github.com/01org/mkl-dnn) installation, please reference [TinyMind's MKL install instructions](https://github.com/mind/wheels#mkl)
