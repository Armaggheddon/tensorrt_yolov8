from tensorrt_yolov8 import TRTYoloV8
from tensorrt_yolov8.task.utils import draw_segmentation_results

import cv2


if __name__ == "__main__":

    model_path = "yolov8s_seg_b1_fp32.engine"
    image_path = "demo_img.jpg"

    classification = TRTYoloV8("segmentation", model_path)

    image = cv2.imread(image_path)
    results = classification(image, min_prob=0.5, top_k=3)

    img_result = draw_segmentation_results(image, results)

    cv2.imwrite(f"{image_path.split('.jpg')[0]}_result.jpg", img_result)