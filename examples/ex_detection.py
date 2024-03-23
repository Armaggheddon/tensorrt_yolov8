from tensorrt_yolov8 import TRTYoloV8
from tensorrt_yolov8.task.utils import draw_detection_results

import cv2


if __name__ == "__main__":


    # model_path = "yolov8s_det_b1_fp32.engine"
    model_path = "yolov8s_b1_fp16_export.engine"
    image_path = "demo_img.jpg"

    detection = TRTYoloV8("detection", model_path)

    image = cv2.imread(image_path)
    results = detection(image, min_prob=0.5, top_k=3)

    img_result = draw_detection_results(image, results)

    cv2.imwrite(f"{image_path.split('.jpg')[0]}_result.jpg", img_result)