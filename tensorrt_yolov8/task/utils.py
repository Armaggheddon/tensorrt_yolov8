import numpy as np
import cv2

# OBB utilities
def get_scaled_obb_bboxes(
        results: list['ObbResult'],
        image_shape: tuple[int, int],
        offset_x: int = 0, offset_y: int = 0
) -> list[cv2.RotatedRect]:
    
    t_h, t_w = image_shape
    bboxes = []

    for r in results:
        rect = r.rotated_rect
        c_x, c_y = rect.center * np.array([t_w, t_h]) + np.array([offset_x, offset_y])
        w, h = rect.size * np.array([t_w, t_h])
        bboxes.append(cv2.RotatedRect((c_x, c_y), (w, h), rect.angle))
    return bboxes

def draw_obb_results(
        image: np.ndarray,
        results: list['ObbResult'],
        draw_label: bool = True,
        draw_confidence:bool=True,
        offset_x : int = 0, offset_y : int = 0,
        scale_x : int = 0, scale_y : int = 0,
        use_copy : bool = True,
) -> np.ndarray:
    
    img_cp = image.copy() if use_copy else image 

    scaled_bboxes = get_scaled_obb_bboxes(
        results, 
        image.shape[:2] if scale_x == 0 or scale_y == 0 else (scale_y, scale_x),
        offset_x, offset_y
        )
    
    for i, r in enumerate(results):
        color = (0, 0, 255)
        rot_rect : cv2.RotatedRect = scaled_bboxes[i]
        points = rot_rect.points()
        for i in range(4):
            cv2.line(img_cp, tuple(map(int, points[i])), tuple(map(int, points[(i+1)%4])), color, 2)
            
        min_id = 0
        if draw_label or draw_confidence:
            # get the point with the min y coordinate
            # so that text is not above the bbox edges
            min_id = np.argmin(points[:, 1], axis=0)
        
        if draw_label:
            cv2.putText(
                img_cp,
                f"{r.class_label}",
                (int(points[min_id, 0]), int(points[min_id, 1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)
        if draw_confidence:
            cv2.putText(
                img_cp,
                f"{r.confidence:.2f}",
                (int(points[min_id][0]), int(points[min_id][1]) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)
    return img_cp


# Segmentation utilities
def get_scaled_segmentation_mask(
        mask : np.ndarray,
        image_shape : tuple[int, int],
        bbox : tuple[int, int, int, int],
        mask_threshold : float = 0.5,
        wrt_original : bool = False,
        mask_inner : bool = True
):
    """
    Args:
    - mask: np.ndarray, mask to be scaled
    - image_shape: tuple[int, int], original image shape as (height, width)
    - bbox: tuple[int, int, int, int], bbox coordinates in 0-1 range as (x1, y1, x2, y2)
    - mask_threshold: float = 0.5, threshold for binarizing the mask
    - wrt_original: bool = False, if True, the final mask is resized to the original image size,
        otherwise, the final mask is resized to the bbox size. Can be useful for object
        extraction from image
    
    Returns:
    - np.ndarray: scaled mask
    """
    h, w = image_shape
    mx1, my1, mx2, my2 = int(bbox[0] * 160), int(bbox[1] * 160), int(bbox[2] * 160), int(bbox[3] * 160)
    bbox_mask = mask[my1:my2, mx1:mx2]

    # get bbox coordinates in the original image scale
    bx1, by1, bx2, by2 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)

    bbox_mask = cv2.resize(bbox_mask, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_LINEAR)
    
    if mask_inner:
        bbox_mask = (bbox_mask > mask_threshold).astype(np.uint8)
    else:
        bbox_mask = (bbox_mask < mask_threshold).astype(np.uint8)

    if wrt_original:
        # is more expensive as the image size is larger
        border = np.zeros((h, w), dtype=np.uint8)
        border[by1:by2, bx1:bx2] = bbox_mask
        bbox_mask = border

    return bbox_mask

def draw_segmentation_results(
        image : np.ndarray, 
        results : list['SegmentationResult'], 
        draw_bbox : bool = True,
        draw_label : bool = True,
        draw_confidence : bool = True,
        mask_threshold : float = 0.5,
        mask_binary : bool = False,
        mask_alpha : float = 0.3,
        offset_x : int = 0, offset_y : int = 0,
        scale_x : int = 0, scale_y : int = 0,
        use_copy : bool = True,
    ) -> np.ndarray:
    """
    Draws the segmentation results on a copy of the given image. Allows for
    scaling and offsetting the bboxes, which can be useful if the image is
    a crop of a larger image, e.g. if an image is processed in tiles and
    the results need to be drawn on the original image can be used as, for each tile:
    `draw_segmentation_results(original_img, results, offset_x=tile_x, offset_y=tile_y, use_copy=False, scale_x=tile_w, scale_y=tile_h)`
    
    See the examples for more details.

    Args:
    - image: np.ndarray, image to draw on
    - results: list[SegmentationResult] -- list of segmentation results. This is obtained from
        the model output when raw_output is False
    - draw_bbox: bool = True -- if True, the bounding box is drawn
    - draw_label: bool = True -- if True, the label is drawn
    - draw_confidence: bool = True -- if True, the confidence score is drawn
    - mask_threshold: float = 0.5 -- threshold for binarizing the mask
    - mask_binary: bool = False -- if True, the mask is drawn as a binary mask, otherwise
        the mask is drawn with alpha overlay
    - mask_alpha: float = 0.3 -- alpha value for the mask overlay, ignored when binary mask is True
    - offset_x: int = 0 -- offset to be added to the x coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    - offset_y: int = 0 -- offset to be added to the y coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    - scale_x: int = 0 -- target x resolution to scale the bboxes to, overrides the image shape
    - scale_y: int = 0 -- target y resolution to scale the bboxes to, overrides the image shape
    - use_copy: bool = True -- if True, the overlay is drawn on a copy of the original image, otherwise

    Returns:
    - np.ndarray: a copy of the original image with the results drawn or the original image if use_copy is False
    """
    img_cp = image.copy() if use_copy else image
    h, w = image.shape[:2] if scale_x == 0 or scale_y == 0 else (scale_y, scale_x)
    masks = get_scaled_segmentation_masks(
        results,
        (h, w),
        mask_threshold=mask_threshold,
        wrt_original=False,
        mask_inner= not mask_binary
    )
    # TODO: use different color for different classes ? maybe user definable?
    color = (0, 0, 255)

    for i, r in enumerate(results):
        mask = masks[i]

        bx1 = int(r.x * w + offset_x)
        by1 = int(r.y * h + offset_y)
        bx2 = int(r.x2 * w + offset_x)
        by2 = int(r.y2 * h + offset_y)

        if draw_bbox:
            cv2.rectangle(
                img_cp,
                (bx1, by1), (bx2, by2),
                color, 2)
        if draw_label:
            cv2.putText(
                img_cp,
                f"{r.class_label}",
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)
        if draw_confidence:
            cv2.putText(
                img_cp,
                f"{r.confidence:.2f}",
                (bx1, by1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)
        
        if mask_binary:
            img_cp[by1:by2, bx1:bx2] = cv2.bitwise_and(
                img_cp[by1:by2, bx1:bx2], img_cp[by1:by2, bx1:bx2], mask=mask)
        else:
            # colored mask with alpha support
            colored_mask = np.moveaxis(
                np.expand_dims(mask, axis=0).repeat(3, axis=0), 0, -1)
            # print(f"Got {colored_mask.shape=}")
            masked = np.ma.MaskedArray(
                img_cp[by1:by2, bx1:bx2],
                mask=colored_mask,
                fill_value=color).filled()
            img_cp[by1:by2, bx1:bx2] = cv2.addWeighted(
                img_cp[by1:by2, bx1:bx2], 1-mask_alpha, masked, mask_alpha, 0)
    return img_cp

def get_scaled_segmentation_masks(
        results : list['SegmentationResult'], 
        image_shape : tuple[int, int], 
        mask_threshold : float = 0.5,
        wrt_original : bool = False,
        mask_inner : bool = True,
        mask_output_width : int = 160,
        mask_output_height : int = 160
    ) -> list[np.ndarray]:
    """
    Args:
    - results: list[SegmentationResult]
    - image_shape: tuple[height: int, width: int] -- original image shape
    - mask_threshold: float = 0.5 -- threshold for binarizing the mask
    - wrt_original: bool = False -- if True, the masks are resized to the original image size,
        otherwise, the masks are resized to the bbox size. Can be useful for object
        extraction from image
    
    Returns:
    - list[np.ndarray]: list of scaled masks
    """
    masks = []
    h, w = image_shape

    for r in results:
        mask = r.mask

        # keep only the mask region that applies to the current box
        # to avoid computing on the whole image scale
        # bbox coordinates are in 0-1 range
        # TODO: 160 should ideally match mask output size...
        mx1 = int(r.x * mask_output_width)
        my1 = int(r.y * mask_output_height)
        mx2 = int(r.x2 * mask_output_width)
        my2 = int(r.y2 * mask_output_height)
        bbox_mask = mask[my1:my2, mx1:mx2]

        # get bbox coordinates in the original image scale
        bx1, by1, bx2, by2 = int(r.x * w), int(r.y * h), int(r.x2 * w), int(r.y2 * h)

        bbox_mask = cv2.resize(bbox_mask, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_LINEAR)
        if mask_inner:
            bbox_mask = (bbox_mask > mask_threshold).astype(np.uint8)
        else:
            bbox_mask = (bbox_mask < mask_threshold).astype(np.uint8)

        if wrt_original:
            border = np.zeros((h, w), dtype=np.uint8)
            border[by1:by2, bx1:bx2] = bbox_mask
            bbox_mask = border

        # bbox_mask is a matrix with 0-1 values where
        # only the inner region of the mask is 1 and
        # outside is 0
        masks.append(bbox_mask)
    return masks

def get_printable_masks(
        results : list['SegmentationResult'],
        image_shape : tuple[int, int] = None,
):
    """
    Args:
    - results: list[SegmentationResult] -- list of segmentation results obtained from the model
        output when raw_output is False
    - image_shape: tuple[height:int, width:int] = None -- target resolution to scale the masks to, if None,
        the masks are not resized and are left of the original size
    
    Returns:
    - list[np.ndarray]: list of masks as numpy arrays ready to be saved as images (single channel uint8 images)
    """
    result = []
    for i, res in enumerate(results):
        if image_shape is not None:
            res.mask = cv2.resize(res.mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
        res.mask *= 255
        result.append(res.mask)
    
    return result


# Detection utilities
def get_scaled_detection_bboxes(
        results : list['DetectionResult'], 
        image_shape : tuple[int, int],
        offset_x : int = 0, offset_y : int = 0,
    ) -> list[tuple[int, int, int, int]]:
    """
    Args:
    - results: list[DetectionResult] -- list of detection results obtained from the model
        output when raw_output is False
    - image_shape: tuple[height: int, width: int] -- target resolution to scale the bboxes to
    - offset_x: int = 0 -- offset to be added to the x coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    - offset_y: int = 0 -- offset to be added to the y coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    
    Returns:
    - list[tuple[int, int, int, int]]: list of scaled bboxes
    """
    h, w = image_shape
    bboxes = []
    for r in results:
        # get bbox coordinates in the original image scale
        bx1 = int(r.x * w + offset_x)
        by1 = int(r.y * h + offset_y)
        bx2 = int(r.x2 * w + offset_x)
        by2 = int(r.y2 * h + offset_y)
        bboxes.append((bx1, by1, bx2, by2))
    return bboxes

def draw_detection_results(
        image : np.ndarray,
        results : list['DetectionResult'],
        draw_label : bool = True,
        draw_confidence : bool = True,
        offset_x : int = 0, offset_y : int = 0,
        scale_x : int = 0, scale_y : int = 0,
        use_copy : bool = True,
) -> np.ndarray:
    """
    Draws the detection results on a copy of the given image. Allows for 
    scaling and offsetting the bboxes, which can be useful if the image is
    a crop of a larger image, e.g. if an image is processed in tiles and 
    the results need to be drawn on the original image can be used as, for each tile:
    `draw_detection_results(original_img, results, offset_x=tile_x, offset_y=tile_y, use_copy=False, scale_x=tile_w, scale_y=tile_h)`

    Args:
    - image: np.ndarray -- image to draw on (this image is not modified)
    - results: list[DetectionResult] -- list of detection results. This is obtained from
        the output when raw_output is False
    - draw_label: bool = True -- if True, the label is drawn
    - draw_confidence: bool = True -- if True, the confidence score is drawn
    - offset_x: int = 0 -- offset to be added to the x coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    - offset_y: int = 0 -- offset to be added to the y coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    - scale_x: int = 0 -- target x resolution to scale the bboxes to, overrides the image shape
    - scale_y: int = 0 -- target y resolution to scale the bboxes to, overrides the image shape
    - use_copy: bool = True -- if True, the overlay is drawn on a copy of the original image, otherwise
        the original image is modified
    
    Returns:
    - np.ndarray: a copy of the original image with the results drawn
    """
    img_cp = image.copy() if use_copy else image 

    scaled_bboxes = get_scaled_detection_bboxes(
        results, 
        image.shape[:2] if scale_x == 0 or scale_y == 0 else (scale_y, scale_x),
        offset_x, offset_y
        )
    for i, r in enumerate(results):
        color = (0, 0, 255)
        bx1, by1, bx2, by2 = scaled_bboxes[i]
        cv2.rectangle(
            img_cp,
            (bx1, by1), (bx2, by2),
            color, 2)
        if draw_label:
            cv2.putText(
                img_cp,
                f"{r.class_label}",
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)
        if draw_confidence:
            cv2.putText(
                img_cp,
                f"{r.confidence:.2f}",
                (bx1, by1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)
    return img_cp


# Pose utilities
def get_scaled_pose_lines_for_skeleton(
        keypoints: dict[str, 'KeyPoint'],
        img_size: tuple[int, int],
        offset_x: int=0, 
        offset_y: int=0):
    """
    Builds a list of pointes representing start and end coordinates
    of the lines that form the skeleton of the pose.

    Args:
    - keypoints: dict[str, KeyPoint] -- dictionary of keypoints, obtained from 
        output[i].keypoints of the model, when raw_output is False, for a given detection
    - img_size: tuple[int, int] -- original image size as (height, width)
    - offset_x: int = 0 -- offset to be added to the x coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    - offset_y: int = 0 -- offset to be added to the y coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    
    Returns:
    - list[tuple[int, int, int, int]] -- list of points as (x1, y1, x2, y2) representing
        the start and end coordinates of the lines of the skeleton
    """
    h, w = img_size
    lines = [
        # face
        (keypoints["right-ear"].x * w + offset_x, keypoints["right-ear"].y * h + offset_y, keypoints["right-eye"].x * w + offset_x, keypoints["right-eye"].y * h + offset_y),
        (keypoints["right-eye"].x * w + offset_x , keypoints["right-eye"].y * h + offset_y, keypoints["nose"].x * w + offset_x, keypoints["nose"].y * h + offset_y),
        (keypoints["nose"].x * w + offset_x, keypoints["nose"].y * h + offset_y, keypoints["left-eye"].x * w + offset_x, keypoints["left-eye"].y * h + offset_y),
        (keypoints["left-eye"].x * w + offset_x, keypoints["left-eye"].y * h + offset_y, keypoints["left-ear"].x * w + offset_x, keypoints["left-ear"].y * h + offset_y),
        # right side
        (keypoints["right-wrist"].x * w + offset_x, keypoints["right-wrist"].y * h + offset_y, keypoints["right-elbow"].x * w + offset_x, keypoints["right-elbow"].y * h + offset_y),
        (keypoints["right-elbow"].x * w + offset_x, keypoints["right-elbow"].y * h + offset_y, keypoints["right-shoulder"].x * w + offset_x, keypoints["right-shoulder"].y * h + offset_y),
        (keypoints["right-shoulder"].x * w + offset_x, keypoints["right-shoulder"].y * h + offset_y, keypoints["right-hip"].x * w + offset_x, keypoints["right-hip"].y * h + offset_y),
        (keypoints["right-hip"].x * w + offset_x, keypoints["right-hip"].y * h + offset_y, keypoints["right-knee"].x * w + offset_x, keypoints["right-knee"].y * h + offset_y),
        (keypoints["right-knee"].x * w + offset_x, keypoints["right-knee"].y * h + offset_y, keypoints["right-ankle"].x * w + offset_x, keypoints["right-ankle"].y * h + offset_y),
        # left side
        (keypoints["left-wrist"].x * w + offset_x, keypoints["left-wrist"].y * h + offset_y, keypoints["left-elbow"].x * w + offset_x, keypoints["left-elbow"].y * h + offset_y),
        (keypoints["left-elbow"].x * w + offset_x, keypoints["left-elbow"].y * h + offset_y, keypoints["left-shoulder"].x * w + offset_x, keypoints["left-shoulder"].y * h + offset_y),
        (keypoints["left-shoulder"].x * w + offset_x, keypoints["left-shoulder"].y * h + offset_y, keypoints["left-hip"].x * w + offset_x, keypoints["left-hip"].y * h + offset_y),
        (keypoints["left-hip"].x * w + offset_x, keypoints["left-hip"].y * h + offset_y, keypoints["left-knee"].x * w + offset_x, keypoints["left-knee"].y * h + offset_y),
        (keypoints["left-knee"].x * w + offset_x, keypoints["left-knee"].y * h + offset_y, keypoints["left-ankle"].x * w + offset_x, keypoints["left-ankle"].y * h + offset_y),

        # connect left and right hips and shoulders
        (keypoints["right-hip"].x * w + offset_x, keypoints["right-hip"].y * h + offset_y, keypoints["left-hip"].x * w + offset_x, keypoints["left-hip"].y * h + offset_y),
        (keypoints["right-shoulder"].x * w + offset_x, keypoints["right-shoulder"].y * h + offset_y, keypoints["left-shoulder"].x * w + offset_x, keypoints["left-shoulder"].y * h + offset_y),
    ]
    return lines

def draw_pose_results(
        image : np.ndarray,
        results : list['PoseResult'],
        draw_bbox : bool = True,
        draw_confidence : bool = True,
        draw_skeleton : bool = True,
        draw_keypoints : bool = True,
        offset_x : int = 0, offset_y : int = 0,
        scale_x : int = 0, scale_y : int = 0,
        use_copy : bool = True,
) -> np.ndarray:
    """
    Draws the pose results on a copy of the given image. Allows for 
    scaling and offsetting the bboxes and keypoints, which can be useful if the image is
    a crop of a larger image, e.g. if an image is processed in tiles and 
    the results need to be drawn on the original image can be used as, for each tile:
    `draw_pose_results(original_img, results, offset_x=tile_x, offset_y=tile_y, use_copy=False, scale_x=tile_w, scale_y=tile_h)`

    Args:
    - image: np.ndarray, image to draw on (this image is not modified)
    - results: list[PoseResult], list of pose results. This is obtained from
        the result of calling the model when raw_output is False
    - draw_bbox: bool = True, if True, the bounding box is drawn (in green)
    - draw_confidence: bool = True, if True, the confidence score is drawn (in green)
    - draw_skeleton: bool = True, if True, the skeleton is drawn (in red)
    - draw_keypoints: bool = True, if True, the keypoints are drawn (in yellow)
    - offset_x: int = 0, offset to be added to the x coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    - offset_y: int = 0, offset to be added to the y coordinate of the keypoints.
        Can be useful if the image is a crop of a larger image
    - scale_x: int = 0, target x resolution to scale the bboxes to, overrides the image shape
    - scale_y: int = 0, target y resolution to scale the bboxes to, overrides the image shape
    - use_copy: bool = True, if True, the overlay is drawn on a copy of the original image, otherwise
        the original image is modified

    Returns:
    - np.ndarray: a copy of the original image with the results drawn
    """
    img_cp = image.copy() if use_copy else image
    h, w = image.shape[:2] if scale_x == 0 or scale_y == 0 else (scale_y, scale_x)
    bbox_color = (0, 255, 0)
    skeleton_color = (0, 0, 255)
    kp_color = (0, 255, 255)
    for r in results:

        bx1 = int(r.x * w + offset_x)
        by1 = int(r.y * h + offset_y)
        bx2 = int(r.x2 * w + offset_x)
        by2 = int(r.y2 * h + offset_y)
        if draw_bbox:
            cv2.rectangle(
                img_cp,
                (bx1, by1), (bx2, by2),
                bbox_color, 2)
        if draw_confidence:
            cv2.putText(
                img_cp,
                f"{r.confidence:.2f}",
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, bbox_color, 2)
        if draw_skeleton:
            lines = get_scaled_pose_lines_for_skeleton(r.keypoints, (h, w), offset_x, offset_y)
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(img_cp, (int(x1), int(y1)), (int(x2), int(y2)), skeleton_color, 2)
        if draw_keypoints:
            for kp in r.keypoints.values():
                cv2.circle(img_cp, (int(kp.x*w), int(kp.y*h)), 2, kp_color, -1)
    return img_cp


# Labels for classes, can be easily obtained as
# label = xxx_labels[class_id]
pose_keypoint_labels = [
        "nose",
        "left-eye",
        "right-eye",
        "left-ear",
        "right-ear",
        "left-shoulder",
        "right-shoulder",
        "left-elbow",
        "right-elbow",
        "left-wrist",
        "right-wrist",
        "left-hip",
        "right-hip",
        "left-knee",
        "right-knee",
        "left-ankle",
        "right-ankle",
    ]

obb_labels = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
]

detection_labels = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",]

classification_labels = [
    "tench",
    "goldfish",
    "great white shark",
    "tiger shark",
    "hammerhead shark",
    "electric ray",
    "stingray",
    "cock",
    "hen",
    "ostrich",
    "brambling",
    "goldfinch",
    "house finch",
    "junco",
    "indigo bunting",
    "American robin",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "American dipper",
    "kite",
    "bald eagle",
    "vulture",
    "great grey owl",
    "fire salamander",
    "smooth newt",
    "newt",
    "spotted salamander",
    "axolotl",
    "American bullfrog",
    "tree frog",
    "tailed frog",
    "loggerhead sea turtle",
    "leatherback sea turtle",
    "mud turtle",
    "terrapin",
    "box turtle",
    "banded gecko",
    "green iguana",
    "Carolina anole",
    "desert grassland whiptail lizard",
    "agama",
    "frilled-necked lizard",
    "alligator lizard",
    "Gila monster",
    "European green lizard",
    "chameleon",
    "Komodo dragon",
    "Nile crocodile",
    "American alligator",
    "triceratops",
    "worm snake",
    "ring-necked snake",
    "eastern hog-nosed snake",
    "smooth green snake",
    "kingsnake",
    "garter snake",
    "water snake",
    "vine snake",
    "night snake",
    "boa constrictor",
    "African rock python",
    "Indian cobra",
    "green mamba",
    "sea snake",
    "Saharan horned viper",
    "eastern diamondback rattlesnake",
    "sidewinder",
    "trilobite",
    "harvestman",
    "scorpion",
    "yellow garden spider",
    "barn spider",
    "European garden spider",
    "southern black widow",
    "tarantula",
    "wolf spider",
    "tick",
    "centipede",
    "black grouse",
    "ptarmigan",
    "ruffed grouse",
    "prairie grouse",
    "peacock",
    "quail",
    "partridge",
    "grey parrot",
    "macaw",
    "sulphur-crested cockatoo",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "duck",
    "red-breasted merganser",
    "goose",
    "black swan",
    "tusker",
    "echidna",
    "platypus",
    "wallaby",
    "koala",
    "wombat",
    "jellyfish",
    "sea anemone",
    "brain coral",
    "flatworm",
    "nematode",
    "conch",
    "snail",
    "slug",
    "sea slug",
    "chiton",
    "chambered nautilus",
    "Dungeness crab",
    "rock crab",
    "fiddler crab",
    "red king crab",
    "American lobster",
    "spiny lobster",
    "crayfish",
    "hermit crab",
    "isopod",
    "white stork",
    "black stork",
    "spoonbill",
    "flamingo",
    "little blue heron",
    "great egret",
    "bittern",
    "crane (bird)",
    "limpkin",
    "common gallinule",
    "American coot",
    "bustard",
    "ruddy turnstone",
    "dunlin",
    "common redshank",
    "dowitcher",
    "oystercatcher",
    "pelican",
    "king penguin",
    "albatross",
    "grey whale",
    "killer whale",
    "dugong",
    "sea lion",
    "Chihuahua",
    "Japanese Chin",
    "Maltese",
    "Pekingese",
    "Shih Tzu",
    "King Charles Spaniel",
    "Papillon",
    "toy terrier",
    "Rhodesian Ridgeback",
    "Afghan Hound",
    "Basset Hound",
    "Beagle",
    "Bloodhound",
    "Bluetick Coonhound",
    "Black and Tan Coonhound",
    "Treeing Walker Coonhound",
    "English foxhound",
    "Redbone Coonhound",
    "borzoi",
    "Irish Wolfhound",
    "Italian Greyhound",
    "Whippet",
    "Ibizan Hound",
    "Norwegian Elkhound",
    "Otterhound",
    "Saluki",
    "Scottish Deerhound",
    "Weimaraner",
    "Staffordshire Bull Terrier",
    "American Staffordshire Terrier",
    "Bedlington Terrier",
    "Border Terrier",
    "Kerry Blue Terrier",
    "Irish Terrier",
    "Norfolk Terrier",
    "Norwich Terrier",
    "Yorkshire Terrier",
    "Wire Fox Terrier",
    "Lakeland Terrier",
    "Sealyham Terrier",
    "Airedale Terrier",
    "Cairn Terrier",
    "Australian Terrier",
    "Dandie Dinmont Terrier",
    "Boston Terrier",
    "Miniature Schnauzer",
    "Giant Schnauzer",
    "Standard Schnauzer",
    "Scottish Terrier",
    "Tibetan Terrier",
    "Australian Silky Terrier",
    "Soft-coated Wheaten Terrier",
    "West Highland White Terrier",
    "Lhasa Apso",
    "Flat-Coated Retriever",
    "Curly-coated Retriever",
    "Golden Retriever",
    "Labrador Retriever",
    "Chesapeake Bay Retriever",
    "German Shorthaired Pointer",
    "Vizsla",
    "English Setter",
    "Irish Setter",
    "Gordon Setter",
    "Brittany",
    "Clumber Spaniel",
    "English Springer Spaniel",
    "Welsh Springer Spaniel",
    "Cocker Spaniels",
    "Sussex Spaniel",
    "Irish Water Spaniel",
    "Kuvasz",
    "Schipperke",
    "Groenendael",
    "Malinois",
    "Briard",
    "Australian Kelpie",
    "Komondor",
    "Old English Sheepdog",
    "Shetland Sheepdog",
    "collie",
    "Border Collie",
    "Bouvier des Flandres",
    "Rottweiler",
    "German Shepherd Dog",
    "Dobermann",
    "Miniature Pinscher",
    "Greater Swiss Mountain Dog",
    "Bernese Mountain Dog",
    "Appenzeller Sennenhund",
    "Entlebucher Sennenhund",
    "Boxer",
    "Bullmastiff",
    "Tibetan Mastiff",
    "French Bulldog",
    "Great Dane",
    "St. Bernard",
    "husky",
    "Alaskan Malamute",
    "Siberian Husky",
    "Dalmatian",
    "Affenpinscher",
    "Basenji",
    "pug",
    "Leonberger",
    "Newfoundland",
    "Pyrenean Mountain Dog",
    "Samoyed",
    "Pomeranian",
    "Chow Chow",
    "Keeshond",
    "Griffon Bruxellois",
    "Pembroke Welsh Corgi",
    "Cardigan Welsh Corgi",
    "Toy Poodle",
    "Miniature Poodle",
    "Standard Poodle",
    "Mexican hairless dog",
    "grey wolf",
    "Alaskan tundra wolf",
    "red wolf",
    "coyote",
    "dingo",
    "dhole",
    "African wild dog",
    "hyena",
    "red fox",
    "kit fox",
    "Arctic fox",
    "grey fox",
    "tabby cat",
    "tiger cat",
    "Persian cat",
    "Siamese cat",
    "Egyptian Mau",
    "cougar",
    "lynx",
    "leopard",
    "snow leopard",
    "jaguar",
    "lion",
    "tiger",
    "cheetah",
    "brown bear",
    "American black bear",
    "polar bear",
    "sloth bear",
    "mongoose",
    "meerkat",
    "tiger beetle",
    "ladybug",
    "ground beetle",
    "longhorn beetle",
    "leaf beetle",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "ant",
    "grasshopper",
    "cricket",
    "stick insect",
    "cockroach",
    "mantis",
    "cicada",
    "leafhopper",
    "lacewing",
    "dragonfly",
    "damselfly",
    "red admiral",
    "ringlet",
    "monarch butterfly",
    "small white",
    "sulphur butterfly",
    "gossamer-winged butterfly",
    "starfish",
    "sea urchin",
    "sea cucumber",
    "cottontail rabbit",
    "hare",
    "Angora rabbit",
    "hamster",
    "porcupine",
    "fox squirrel",
    "marmot",
    "beaver",
    "guinea pig",
    "common sorrel",
    "zebra",
    "pig",
    "wild boar",
    "warthog",
    "hippopotamus",
    "ox",
    "water buffalo",
    "bison",
    "ram",
    "bighorn sheep",
    "Alpine ibex",
    "hartebeest",
    "impala",
    "gazelle",
    "dromedary",
    "llama",
    "weasel",
    "mink",
    "European polecat",
    "black-footed ferret",
    "otter",
    "skunk",
    "badger",
    "armadillo",
    "three-toed sloth",
    "orangutan",
    "gorilla",
    "chimpanzee",
    "gibbon",
    "siamang",
    "guenon",
    "patas monkey",
    "baboon",
    "macaque",
    "langur",
    "black-and-white colobus",
    "proboscis monkey",
    "marmoset",
    "white-headed capuchin",
    "howler monkey",
    "titi",
    "Geoffroy's spider monkey",
    "common squirrel monkey",
    "ring-tailed lemur",
    "indri",
    "Asian elephant",
    "African bush elephant",
    "red panda",
    "giant panda",
    "snoek",
    "eel",
    "coho salmon",
    "rock beauty",
    "clownfish",
    "sturgeon",
    "garfish",
    "lionfish",
    "pufferfish",
    "abacus",
    "abaya",
    "academic gown",
    "accordion",
    "acoustic guitar",
    "aircraft carrier",
    "airliner",
    "airship",
    "altar",
    "ambulance",
    "amphibious vehicle",
    "analog clock",
    "apiary",
    "apron",
    "waste container",
    "assault rifle",
    "backpack",
    "bakery",
    "balance beam",
    "balloon",
    "ballpoint pen",
    "Band-Aid",
    "banjo",
    "baluster",
    "barbell",
    "barber chair",
    "barbershop",
    "barn",
    "barometer",
    "barrel",
    "wheelbarrow",
    "baseball",
    "basketball",
    "bassinet",
    "bassoon",
    "swimming cap",
    "bath towel",
    "bathtub",
    "station wagon",
    "lighthouse",
    "beaker",
    "military cap",
    "beer bottle",
    "beer glass",
    "bell-cot",
    "bib",
    "tandem bicycle",
    "bikini",
    "ring binder",
    "binoculars",
    "birdhouse",
    "boathouse",
    "bobsleigh",
    "bolo tie",
    "poke bonnet",
    "bookcase",
    "bookstore",
    "bottle cap",
    "bow",
    "bow tie",
    "brass",
    "bra",
    "breakwater",
    "breastplate",
    "broom",
    "bucket",
    "buckle",
    "bulletproof vest",
    "high-speed train",
    "butcher shop",
    "taxicab",
    "cauldron",
    "candle",
    "cannon",
    "canoe",
    "can opener",
    "cardigan",
    "car mirror",
    "carousel",
    "tool kit",
    "carton",
    "car wheel",
    "automated teller machine",
    "cassette",
    "cassette player",
    "castle",
    "catamaran",
    "CD player",
    "cello",
    "mobile phone",
    "chain",
    "chain-link fence",
    "chain mail",
    "chainsaw",
    "chest",
    "chiffonier",
    "chime",
    "china cabinet",
    "Christmas stocking",
    "church",
    "movie theater",
    "cleaver",
    "cliff dwelling",
    "cloak",
    "clogs",
    "cocktail shaker",
    "coffee mug",
    "coffeemaker",
    "coil",
    "combination lock",
    "computer keyboard",
    "confectionery store",
    "container ship",
    "convertible",
    "corkscrew",
    "cornet",
    "cowboy boot",
    "cowboy hat",
    "cradle",
    "crane (machine)",
    "crash helmet",
    "crate",
    "infant bed",
    "Crock Pot",
    "croquet ball",
    "crutch",
    "cuirass",
    "dam",
    "desk",
    "desktop computer",
    "rotary dial telephone",
    "diaper",
    "digital clock",
    "digital watch",
    "dining table",
    "dishcloth",
    "dishwasher",
    "disc brake",
    "dock",
    "dog sled",
    "dome",
    "doormat",
    "drilling rig",
    "drum",
    "drumstick",
    "dumbbell",
    "Dutch oven",
    "electric fan",
    "electric guitar",
    "electric locomotive",
    "entertainment center",
    "envelope",
    "espresso machine",
    "face powder",
    "feather boa",
    "filing cabinet",
    "fireboat",
    "fire engine",
    "fire screen sheet",
    "flagpole",
    "flute",
    "folding chair",
    "football helmet",
    "forklift",
    "fountain",
    "fountain pen",
    "four-poster bed",
    "freight car",
    "French horn",
    "frying pan",
    "fur coat",
    "garbage truck",
    "gas mask",
    "gas pump",
    "goblet",
    "go-kart",
    "golf ball",
    "golf cart",
    "gondola",
    "gong",
    "gown",
    "grand piano",
    "greenhouse",
    "grille",
    "grocery store",
    "guillotine",
    "barrette",
    "hair spray",
    "half-track",
    "hammer",
    "hamper",
    "hair dryer",
    "hand-held computer",
    "handkerchief",
    "hard disk drive",
    "harmonica",
    "harp",
    "harvester",
    "hatchet",
    "holster",
    "home theater",
    "honeycomb",
    "hook",
    "hoop skirt",
    "horizontal bar",
    "horse-drawn vehicle",
    "hourglass",
    "iPod",
    "clothes iron",
    "jack-o'-lantern",
    "jeans",
    "jeep",
    "T-shirt",
    "jigsaw puzzle",
    "pulled rickshaw",
    "joystick",
    "kimono",
    "knee pad",
    "knot",
    "lab coat",
    "ladle",
    "lampshade",
    "laptop computer",
    "lawn mower",
    "lens cap",
    "paper knife",
    "library",
    "lifeboat",
    "lighter",
    "limousine",
    "ocean liner",
    "lipstick",
    "slip-on shoe",
    "lotion",
    "speaker",
    "loupe",
    "sawmill",
    "magnetic compass",
    "mail bag",
    "mailbox",
    "tights",
    "tank suit",
    "manhole cover",
    "maraca",
    "marimba",
    "mask",
    "match",
    "maypole",
    "maze",
    "measuring cup",
    "medicine chest",
    "megalith",
    "microphone",
    "microwave oven",
    "military uniform",
    "milk can",
    "minibus",
    "miniskirt",
    "minivan",
    "missile",
    "mitten",
    "mixing bowl",
    "mobile home",
    "Model T",
    "modem",
    "monastery",
    "monitor",
    "moped",
    "mortar",
    "square academic cap",
    "mosque",
    "mosquito net",
    "scooter",
    "mountain bike",
    "tent",
    "computer mouse",
    "mousetrap",
    "moving van",
    "muzzle",
    "nail",
    "neck brace",
    "necklace",
    "nipple",
    "notebook computer",
    "obelisk",
    "oboe",
    "ocarina",
    "odometer",
    "oil filter",
    "organ",
    "oscilloscope",
    "overskirt",
    "bullock cart",
    "oxygen mask",
    "packet",
    "paddle",
    "paddle wheel",
    "padlock",
    "paintbrush",
    "pajamas",
    "palace",
    "pan flute",
    "paper towel",
    "parachute",
    "parallel bars",
    "park bench",
    "parking meter",
    "passenger car",
    "patio",
    "payphone",
    "pedestal",
    "pencil case",
    "pencil sharpener",
    "perfume",
    "Petri dish",
    "photocopier",
    "plectrum",
    "Pickelhaube",
    "picket fence",
    "pickup truck",
    "pier",
    "piggy bank",
    "pill bottle",
    "pillow",
    "ping-pong ball",
    "pinwheel",
    "pirate ship",
    "pitcher",
    "hand plane",
    "planetarium",
    "plastic bag",
    "plate rack",
    "plow",
    "plunger",
    "Polaroid camera",
    "pole",
    "police van",
    "poncho",
    "billiard table",
    "soda bottle",
    "pot",
    "potter's wheel",
    "power drill",
    "prayer rug",
    "printer",
    "prison",
    "projectile",
    "projector",
    "hockey puck",
    "punching bag",
    "purse",
    "quill",
    "quilt",
    "race car",
    "racket",
    "radiator",
    "radio",
    "radio telescope",
    "rain barrel",
    "recreational vehicle",
    "reel",
    "reflex camera",
    "refrigerator",
    "remote control",
    "restaurant",
    "revolver",
    "rifle",
    "rocking chair",
    "rotisserie",
    "eraser",
    "rugby ball",
    "ruler",
    "running shoe",
    "safe",
    "safety pin",
    "salt shaker",
    "sandal",
    "sarong",
    "saxophone",
    "scabbard",
    "weighing scale",
    "school bus",
    "schooner",
    "scoreboard",
    "CRT screen",
    "screw",
    "screwdriver",
    "seat belt",
    "sewing machine",
    "shield",
    "shoe store",
    "shoji",
    "shopping basket",
    "shopping cart",
    "shovel",
    "shower cap",
    "shower curtain",
    "ski",
    "ski mask",
    "sleeping bag",
    "slide rule",
    "sliding door",
    "slot machine",
    "snorkel",
    "snowmobile",
    "snowplow",
    "soap dispenser",
    "soccer ball",
    "sock",
    "solar thermal collector",
    "sombrero",
    "soup bowl",
    "space bar",
    "space heater",
    "space shuttle",
    "spatula",
    "motorboat",
    "spider web",
    "spindle",
    "sports car",
    "spotlight",
    "stage",
    "steam locomotive",
    "through arch bridge",
    "steel drum",
    "stethoscope",
    "scarf",
    "stone wall",
    "stopwatch",
    "stove",
    "strainer",
    "tram",
    "stretcher",
    "couch",
    "stupa",
    "submarine",
    "suit",
    "sundial",
    "sunglass",
    "sunglasses",
    "sunscreen",
    "suspension bridge",
    "mop",
    "sweatshirt",
    "swimsuit",
    "swing",
    "switch",
    "syringe",
    "table lamp",
    "tank",
    "tape player",
    "teapot",
    "teddy bear",
    "television",
    "tennis ball",
    "thatched roof",
    "front curtain",
    "thimble",
    "threshing machine",
    "throne",
    "tile roof",
    "toaster",
    "tobacco shop",
    "toilet seat",
    "torch",
    "totem pole",
    "tow truck",
    "toy store",
    "tractor",
    "semi-trailer truck",
    "tray",
    "trench coat",
    "tricycle",
    "trimaran",
    "tripod",
    "triumphal arch",
    "trolleybus",
    "trombone",
    "tub",
    "turnstile",
    "typewriter keyboard",
    "umbrella",
    "unicycle",
    "upright piano",
    "vacuum cleaner",
    "vase",
    "vault",
    "velvet",
    "vending machine",
    "vestment",
    "viaduct",
    "violin",
    "volleyball",
    "waffle iron",
    "wall clock",
    "wallet",
    "wardrobe",
    "military aircraft",
    "sink",
    "washing machine",
    "water bottle",
    "water jug",
    "water tower",
    "whiskey jug",
    "whistle",
    "wig",
    "window screen",
    "window shade",
    "Windsor tie",
    "wine bottle",
    "wing",
    "wok",
    "wooden spoon",
    "wool",
    "split-rail fence",
    "shipwreck",
    "yawl",
    "yurt",
    "website",
    "comic book",
    "crossword",
    "traffic sign",
    "traffic light",
    "dust jacket",
    "menu",
    "plate",
    "guacamole",
    "consomme",
    "hot pot",
    "trifle",
    "ice cream",
    "ice pop",
    "baguette",
    "bagel",
    "pretzel",
    "cheeseburger",
    "hot dog",
    "mashed potato",
    "cabbage",
    "broccoli",
    "cauliflower",
    "zucchini",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber",
    "artichoke",
    "bell pepper",
    "cardoon",
    "mushroom",
    "Granny Smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard apple",
    "pomegranate",
    "hay",
    "carbonara",
    "chocolate syrup",
    "dough",
    "meatloaf",
    "pizza",
    "pot pie",
    "burrito",
    "red wine",
    "espresso",
    "cup",
    "eggnog",
    "alp",
    "bubble",
    "cliff",
    "coral reef",
    "geyser",
    "lakeshore",
    "promontory",
    "shoal",
    "seashore",
    "valley",
    "volcano",
    "baseball player",
    "bridegroom",
    "scuba diver",
    "rapeseed",
    "daisy",
    "yellow lady's slipper",
    "corn",
    "acorn",
    "rose hip",
    "horse chestnut seed",
    "coral fungus",
    "agaric",
    "gyromitra",
    "stinkhorn mushroom",
    "earth star",
    "hen-of-the-woods",
    "bolete",
    "ear",
    "toilet paper",
]