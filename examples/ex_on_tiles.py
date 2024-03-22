from tensorrt_yolov8 import TRTYoloV8
from tensorrt_yolov8.task.utils import (
    draw_detection_results, 
    draw_segmentation_results,
    draw_pose_results
)

import cv2
import numpy as np

def get_img_tiles(image, tile_size=(640, 640), center_pad=True):

    img_tiles = []
    h, w, _ = image.shape

    if w < tile_size[0] or h < tile_size[1]:
        raise ValueError("Tile size must be smaller than image size")

    dx, rx = divmod(w, tile_size[0])
    dy, ry = divmod(h, tile_size[1])

    out_img_x = ((dx + 1) if rx else dx) * tile_size[0]
    out_img_y = ((dy + 1) if ry else dy) * tile_size[1]

    pad_img = np.zeros((out_img_y, out_img_x, 3), dtype=np.uint8)

    delta_x = tile_size[0] - rx
    delta_y = tile_size[1] - ry

    delta_xL = 0
    delta_yT = 0
    if center_pad:
        delta_xL = delta_x//2 + delta_x % 2
        delta_yT = delta_y//2 + delta_y % 2
    
    pad_img[delta_yT:delta_yT+h, delta_xL:delta_xL+w] = image

    #get the tiles
    for off_y in range(0, out_img_y, tile_size[1]):
        for off_x in range(0, out_img_x, tile_size[0]):
            img_tiles.append(pad_img[off_y:off_y+tile_size[1], off_x:off_x+tile_size[0]])
    
    return img_tiles, out_img_x, out_img_y, delta_xL, delta_yT


if __name__ == "__main__":
    """
    Example to demonstrate how to use the TRTYoloV8 class for detection task 
    with image tiling. 
    """
    det_model_path = "yolov8s_det_b1_fp32.engine"
    seg_model_path = "yolov8s_seg_b1_fp32.engine"
    pose_model_path = "yolov8s_pose_b1_fp32.engine"

    image_path = "demo_img.jpg"

    # load the models
    detection = TRTYoloV8("detection", det_model_path)
    segmentation = TRTYoloV8("segmentation", seg_model_path)
    pose = TRTYoloV8("pose", pose_model_path)
     
    # load the image
    image = cv2.imread(image_path)
    detection_result = image.copy()
    segmentation_result = image.copy()
    pose_result = image.copy()

    # set the tile size
    tile_size = (640, 640)
    
    # splits the image into tiles of the specified size,
    # and pads the image with a black border if necessary.
    # By default the image is center padded
    img_tiles, out_img_x, out_img_y, delta_xL, delta_yT = get_img_tiles(image, tile_size)
    x_tiles = out_img_x // tile_size[0] # number of tiles in x direction
    #y_tiles = out_img_y // tile_size[1]

    for i, tile in enumerate(img_tiles):
        det_results = detection(tile, min_prob=0.5, top_k=3)
        seg_results = segmentation(tile, min_prob=0.5, top_k=3)
        pose_results = pose(tile, min_prob=0.5, top_k=3)

        # calculate the offset for the tile with respect to the original image
        off_x = int((i % x_tiles) * tile.shape[1]) - delta_xL
        off_y = int((i // x_tiles) * tile.shape[0]) - delta_yT
        
        detection_result = draw_detection_results(
            detection_result, 
            det_results, 
            offset_x=off_x, 
            offset_y=off_y, 
            scale_x=tile.shape[1], 
            scale_y=tile.shape[0], 
            use_copy=False
        )

        segmentation_result = draw_segmentation_results(
            segmentation_result, 
            seg_results, 
            offset_x=off_x, 
            offset_y=off_y, 
            scale_x=tile.shape[1], 
            scale_y=tile.shape[0], 
            use_copy=False
        )

        pose_result = draw_pose_results(
            pose_result, 
            pose_results, 
            offset_x=off_x, 
            offset_y=off_y, 
            scale_x=tile.shape[1], 
            scale_y=tile.shape[0], 
            use_copy=False
        )

    cv2.imwrite(f"{image_path.split('.jpg')[0]}__tile_det_result.jpg", detection_result)
    cv2.imwrite(f"{image_path.split('.jpg')[0]}__tile_seg_result.jpg", segmentation_result)
    cv2.imwrite(f"{image_path.split('.jpg')[0]}__tile_pose_result.jpg", pose_result)