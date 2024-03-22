from tensorrt_yolov8 import TRTYoloV8
from tensorrt_yolov8.task.utils import draw_segmentation_results, get_scaled_segmentation_masks, get_printable_masks

import cv2

if __name__ == "__main__":
    model_path = "yolov8s_seg_b1_fp32.engine"
    image_path = "demo_img.jpg"

    classification = TRTYoloV8("segmentation", model_path)

    image = cv2.imread(image_path)
    results = classification(image, min_prob=0.5, top_k=3)

    # Obtain the per-object segmentation mask with respect to the original image
    masks = get_scaled_segmentation_masks(results, image.shape[:2], wrt_original=True)
    
    # Obtain the printable model output segmentation mask
    model_masks = get_printable_masks(results, image.shape[:2])

    # Print the segmentation mask as yolo outputs them
    for i, mask in enumerate(model_masks):
        cv2.imwrite(f"{image_path.split('.jpg')[0]}_mask{i}_{results[i].class_label}.jpg", mask)
    

    


    