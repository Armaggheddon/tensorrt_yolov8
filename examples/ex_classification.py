import cv2
from tensorrt_yolov8 import Pipeline


if __name__ == "__main__":

    model_path = "/home/Documents/Experiments/TENSORRT/tensorrt_yolov8/examples/yolov8s_cls_b1_fp16.engine"
    image_path = "/home/Documents/Experiments/TENSORRT/tensorrt_yolov8/examples/demo_img.jpg"
    image = cv2.imread(image_path)


    cls_pipe = Pipeline("classification", model_path)

    results = cls_pipe(image, min_prob=0.4, top_k=3)
    img_out = cls_pipe.draw_results(image, results)

    cv2.imwrite(
        f"{image_path.split('.jpg')[0]}_cls.jpg", 
        img_out
    )

    for res in results:
        print(f"[{res.label_id}] {res.label_name} @ {res.confidence:.2f}")


