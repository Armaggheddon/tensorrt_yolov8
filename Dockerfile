FROM nvcr.io/nvidia/tensorrt:24.01-py3

RUN apt update
RUN pip install git+

RUN python "from tensorrt_yolov8 import TRTYoloV8"
CMD ["/bin/bash"]