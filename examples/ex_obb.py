import cv2
from tensorrt_yolov8 import TRTYoloV8
from tensorrt_yolov8.task.utils import draw_obb_results


if __name__ == "__main__":
    
    model = "yolov8s_obb_b1_fp32.engine"
    image_path = "dock.jpg"
    
    obb = TRTYoloV8("obb", model)

    image = cv2.imread(image_path)
    output = obb(image, min_prob=0.4)
    out_img = draw_obb_results(image, output)

    cv2.imwrite(f"{image_path.split('.jpg')[0]}_result.jpg", out_img)
    