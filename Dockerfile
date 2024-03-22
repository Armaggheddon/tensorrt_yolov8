FROM nvcr.io/nvidia/tensorrt:24.01-py3

RUN apt update
RUN pip install git+https://github.com/Armaggheddon/tensorrt_yolov8.git --no-cache-dir
RUN bash -c python3 -c "from tensorrt_yolov8 import TRTYoloV8;print(TRTYoloV8.available_models())"

CMD ["/bin/bash"]