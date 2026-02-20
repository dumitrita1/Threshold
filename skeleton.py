import cv2
import numpy as np
from numpy import ndarray  # n-dimensional array (real array as contiguous memory of fixed size)


def read_image(path: str) -> ndarray:
    grayscale_image: ndarray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if grayscale_image is None:
        raise RuntimeError(f'could not read image {path}')

    return grayscale_image.astype(np.float32) / 255.0


def write_image(path: str, grayscale_image: ndarray) -> None:
    if grayscale_image.ndim != 2:
        raise ValueError('grayscale_image must be a 2D array')

    image_clipped: ndarray = np.clip(grayscale_image, 0.0, 1.0)
    image_uint8: ndarray = (image_clipped * 255.0).astype(np.uint8)

    cv2.imwrite(path, image_uint8)


def threshold(grayscale_image: ndarray, threshold: float) -> ndarray:
    output: ndarray = np.zeros_like(grayscale_image)

    if 0.0 > threshold > 1.0:
        raise ValueError(f'threshold must be between 0.0 and 1.0, but is {threshold}')
    height, width = grayscale_image.shape
    for y in range(height):
        for x in range(width):
            output[y, x] = 0.0 if grayscale_image[y, x] < threshold else 1.0

    return output


if __name__ == '__main__':
    image: ndarray = read_image('Elfers-Juergen.jpg')
    thresholded_image = threshold(image, 0.5)
    write_image('Elfers-Juergen-Thresholded.jpg', thresholded_image)
