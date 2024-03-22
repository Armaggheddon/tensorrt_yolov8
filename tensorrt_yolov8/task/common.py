import numpy as np
import cv2

def preprocess(images : np.ndarray | list[np.ndarray], input_shape : list[np.ndarray], swap_rb : bool = False) -> np.ndarray:
    """
    Arguments:
    - images -- list of images to preprocess
    - swap_rb -- if the R and B channels should be swapped

    Returns:
    - images -- preprocessed images as a 1D array
    """

    size_x, size_y = input_shape[0][2:]

    def base_transform(image : np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (size_x, size_y)).astype(np.float32)
        if swap_rb:
            image = image[:, :, ::-1] # BGR to RGB
        image /= 255.0
        image = image.transpose((2, 0, 1)) # HWC to CHW
        return np.ascontiguousarray(image).ravel()

    if isinstance(images, np.ndarray):
        images = base_transform(images)
    else:
        for i, image in enumerate(images):
            images[i] = base_transform(image)
        images = np.array(images).ravel()

    return images