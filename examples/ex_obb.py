import cv2
from tensorrt_yolov8 import Pipeline


if __name__ == "__main__":


    # model_path = "yolov8s_det_b1_fp32.engine"
    model_path = "/home/Documents/Experiments/TENSORRT/tensorrt_yolov8/examples/yolov8s_obb_b1_fp32.engine"
    image_path = "/home/Documents/Experiments/TENSORRT/tensorrt_yolov8/examples/dock.jpg"
    image = cv2.imread(image_path)

    det_pipe = Pipeline("obb", model_path)

    results = det_pipe(image, min_prob=0.5, top_k=3)

    img_result = det_pipe.draw_results(image, results)

    cv2.imwrite(f"{image_path.split('.jpg')[0]}_obb.jpg", img_result)