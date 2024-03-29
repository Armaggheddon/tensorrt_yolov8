from tensorrt_yolov8 import EngineHelper
from tensorrt_yolov8.models.utils import draw_pose_results

import cv2


if __name__ == "__main__":


    model_path = "yolov8s_pose_b1_fp32.engine"
    image_path = "demo_img.jpg"

    pose = EngineHelper("pose", model_path)
  
    image = cv2.imread(image_path)
    results = pose(image, min_prob=0.5, top_k=3)

    img_result = draw_pose_results(image, results)

    cv2.imwrite(f"{image_path.split('.jpg')[0]}_result.jpg", img_result)