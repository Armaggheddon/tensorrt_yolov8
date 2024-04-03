from typing import List
import numpy as np
import cv2

def yolo_preprocess(
        *images : np.ndarray, 
        to_shape : np.ndarray,
        swap_rb : bool = False) -> np.ndarray:
    """
    Arguments:
    - images -- list of images to preprocess
    - swap_rb -- if the R and B channels should be swapped

    Returns:
    - images -- preprocessed images as a 1D array
    """

    if len(images) == 0 or len(images) > to_shape[0]:
            raise ValueError(f"The number of images provided is too large: got {len(images)} images, expected at most {to_shape[0]} images.")
    
    # resize to shape x, y
    preproc_images = [cv2.resize(img, (to_shape[2], to_shape[3])).astype(np.float32) for img in images]
    preproc_images = np.array(preproc_images, dtype=np.float32)
    if swap_rb:
        preproc_images = preproc_images[:, :, :, ::-1] # BGR to RGB
    preproc_images /= 255.0
    preproc_images = preproc_images.transpose((0, 3, 1, 2)) # BHWC to BCHW
    
    # Pad batch with zeros if necessary
    if preproc_images.shape[0] < to_shape[0]:
        preproc_images = np.concatenate([preproc_images, np.zeros((to_shape[0] - preproc_images.shape[0], *preproc_images.shape[1:]), dtype=np.float32)], axis=0)
    
    preproc_images = np.ascontiguousarray(np.asarray(preproc_images)).ravel()
    return preproc_images