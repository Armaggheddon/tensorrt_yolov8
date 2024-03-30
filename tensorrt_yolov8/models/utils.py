from typing import List
import numpy as np
import cv2

# OBB utilities
def get_scaled_obb_bboxes(
        results: List['ObbResult'],
        image_shape: tuple[int, int],
        offset_x: int = 0, offset_y: int = 0
) -> List[cv2.RotatedRect]:
    
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
        results: List['ObbResult'],
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
        results : List['SegmentationResult'], 
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
        results : List['SegmentationResult'], 
        image_shape : tuple[int, int], 
        mask_threshold : float = 0.5,
        wrt_original : bool = False,
        mask_inner : bool = True,
        mask_output_width : int = 160,
        mask_output_height : int = 160
    ) -> List[np.ndarray]:
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
        results : List['SegmentationResult'],
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
        results : List['DetectionResult'], 
        image_shape : tuple[int, int],
        offset_x : int = 0, offset_y : int = 0,
    ) -> List[tuple[int, int, int, int]]:
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
        results : List['DetectionResult'],
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
        results : List['PoseResult'],
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
