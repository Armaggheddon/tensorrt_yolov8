# tensorrt_yolov8

Tired of long inference times with your favourite yolov8 models? 

Then this library is for you! Run classification, detection, pose and segmentation yolov8 models as engines by using Nvidia's tensorrt. Seamlessly obtain results or even draw the result overlay on top of the image with just a couple of lines of code.

# Index

- [Requirements](#requirements)
- [Installation](#Installation)
    - [Docker](#docker)
    - [Pip](#pip)
- [Usage](#usage)

# Requirements
This library makes use of Nvidia's specific features, therefore a Nvidia GPU is required. Additionally a working installation of tensorrt is required ([Link](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)). The easiest way to install/use the library is by using the Nvidia's TensorRT docker image available at Nvidia NGC ([Link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt), the library has been tested on the tensorrt:24.01-py3 image)

# Installation

## Docker

- If not already done, configure docker to use the Nvidia runtime as default by editing the file in `/etc/docker/daemon.json`, adding to the first line ([Link](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html#:~:text=Use%20docker%20run%20with%20nvidia,file%20as%20the%20first%20entry.&text=You%20can%20then%20use%20docker%20run%20to%20run%20GPU%2Daccelerated%20containers.)):

    `"default-runtime": "nvidia"`

- Copy the content of `Dockerfile` file and build the image with the command:
    ```bash
    $ docker build -t image:tag .
    ```

- Run the image with the command:
    ```bash
    $ docker run -it image:tag
    ```

## Pip

- Clone the repository to your local machine and run the following commands:
    ```bash
    $ git clone https://github.com/Armaggheddon/tensorrt_yolov8.git
    $ cd tensorrt_yolov8
    $ pip install -U .
    ```

- Alternatively, directly get the library with pip with the command:
    ```bash
    $ pip install git+https://github.com/Armaggheddon/tensorrt_yolov8.git
    ```

- Uninstall the library with:
    ```bash
    $ pip uninstall tensorrt_yolov8
    ```

# Usage