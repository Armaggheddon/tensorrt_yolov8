# tensorrt_yolov8 🏎️💨

Tired of long inference times with your favourite yolov8 models? 

Then this library is for you! Run yolov8's classification, detection, pose and segmentation models as engines by using Nvidia's tensorrt. Seamlessly obtain results or even draw the result overlay on top of the image with just a couple of lines of code.

| ![Results example](/examples/example_results.png) |
|:--:|
| *Example result overlay for detection, pose and segmentation ([image source](https://unsplash.com/it/foto/persone-vicino-a-castle-IU8E-824a-s))* |

| ![Mask results example](/examples/example_mask_results.png) |
|:--:|
| *Example result of the masks obtained from the segmentation model for each object (see [ex_draw_seg_mask.py](/examples/ex_draw_seg_mask.py))* |

# Index

- [Requirements](#requirements)
- [Installation](#installation)
    - [With Docker](#with-docker)
    - [With pip](#with-pip)
- [Sample usage](#sample-usage)
- [Todos](#todos)

# Requirements
This library makes use of Nvidia's specific features, therefore a Nvidia GPU is required. Additionally a working installation of tensorrt is required ([Link](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)). 

# Installation

The easiest way to install/use the library is by using the Nvidia's TensorRT docker image available at Nvidia NGC ([Link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt), the library has been tested on the tensorrt:24.01-py3 image). 

### With Docker

1. If not already done, configure docker to use the Nvidia runtime as default by editing the file in `/etc/docker/daemon.json`, add to the first line ([Link](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html#:~:text=Use%20docker%20run%20with%20nvidia,file%20as%20the%20first%20entry.&text=You%20can%20then%20use%20docker%20run%20to%20run%20GPU%2Daccelerated%20containers.)) `"default-runtime": "nvidia"`

1. Copy the content of `Dockerfile` file and build the image with the command:
    ```bash
    $ docker build -t image:tag .
    ```

1. Run the image with the command:
    ```bash
    $ docker run -it image:tag
    ```

### With pip

1. Clone the repository to your local machine and run the following commands:
    ```bash
    $ git clone https://github.com/Armaggheddon/tensorrt_yolov8.git
    $ cd tensorrt_yolov8
    $ pip install -U .
    ```

1. Alternatively, directly get the library with pip with the command:
    ```bash
    $ pip install git+https://github.com/Armaggheddon/tensorrt_yolov8.git
    ```

1. Uninstall the library with:
    ```bash
    $ pip uninstall tensorrt_yolov8
    ```

# Sample usage

1. Obtain the ONNX file of the desired yolo model. This can be easily done by using Ultralytics library ([Link](https://github.com/ultralytics/ultralytics)). For example the following commands installs, downloads and exports the yolov8s detection model in the current path (See [Link](https://docs.ultralytics.com/it/models/yolov8/#supported-tasks-and-modes) for a list of available model types):
    ```bash 
    $ pip install ultralytics
    $ yolo export model=yolov8s.pt format=onnx
    ```

2. Convert the ONNX model to an Nvidia engine. This can be done using the utility `trtexec` (generally located at `/usr/src/tensorrt/bin/trtexec`) or by using the utility function available in this library with:
    ```python
    from tensorrt_yolov8.utils import engine_builder

    engine_builder.build_engine_from_onnx(
        "path/to/onnx/model.onnx",
        "path/to/created/model.engine"
    )
    ```
    This by default exports yolov8s using FP32 and with batch size=1. This operation is required only the first time. The same model engine can then be used multiple times. If the tensorrt version on which the model has been built is different from the one used to run the engine, the library will complain about this. Fix the issue using the above piece of code.

3. Run the exported model and perform inference with 
    ```python
    import cv2

    from tensorrt_yolov8 import TRTYoloV8
    from tensorrt_yolov8.task.utils import draw_detection_results

    detection = TRTYoloV8("detection", "path/to/model.engine")

    img = cv2.imread("img.jpg")
    results = detection(img, min_prob=0.5)
    img_result = draw_detection_results(img, results)

    cv2.imwrite("result.jpg", img_result)
    ```

4. For additional examples see the files in [Examples](/examples)

# Todos

- [ ] Support for batch sizes greater than 1
- [ ] Support for Yolo 8.1 OBB (Oriented Bounding Box) (?)