from tensorrt_yolov8 import TRTYoloV8

import cv2

if __name__ == "__main__":

    model_path = "yolov8s_cls_b1_fp32.engine"
    image_path = "demo_img.jpg"

    classification = TRTYoloV8("classification", model_path)

    image = cv2.imread(image_path)
    results = classification(image, min_prob=0.5, top_k=3)

    for result in results:
        print(f"[{result.class_id}] {result.class_label} @ {result.probability:.2f}")


