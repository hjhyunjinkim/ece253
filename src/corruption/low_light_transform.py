import os
import cv2
import numpy as np

from low_light_restoration import process_image

def apply_low_light_degradation(image, intensity_factor=0.3, noise_sigma=25):
    """
    Applies low-light degradation to an image by reducing intensity 
    and adding Gaussian noise.

    Args:
        image (numpy.ndarray): The input RGB or BGR image.
        intensity_factor (float): Factor to reduce brightness (0.0 to 1.0).
        noise_sigma (float): Standard deviation of the Gaussian noise.
                             Higher values = grainier image.

    Returns:
        numpy.ndarray: The degraded image in uint8 format.
    """
    # Intensity Reduction
    image_reduced = image.astype(np.float32) * intensity_factor

    # Additive Gaussian Noise
    row, col, ch = image_reduced.shape
    gaussian_noise = np.random.normal(0, noise_sigma, (row, col, ch))
    noisy_image = image_reduced + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Check for clipping
    
    return noisy_image.astype(np.uint8)     # return in 8-byte format


def apply_low_light_poisson_gaussian(image, noise_sigma=25., gamma=2.0, gain=0.5):
    img = image.astype(np.float32) / 255.0

    # Non-linear darkening (gamma correction)
    img = np.power(img, gamma)

    # Linear darkening (exposure reduction)
    img = img * gain

    # Noise
    # Photon count for exposure, 10000.0 for daylight conditions and 1000.0 for low-light conditions (5000.0 for pure white pixel)
    photon_scale = 1000.0
    img = img * photon_scale

    # Poisson noise
    img = np.random.poisson(lam=img).astype(np.float32)

    # Gaussian noise
    awgn = np.random.normal(0, noise_sigma, img.shape)
    img = img + awgn

    # Process back to image
    img = img / photon_scale
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    return img


def degrade_lowlight_image(path_image, path_output, intensity_factor=0.3, noise_sigma=25, gamma=2.0):
    """
    Apply low-light degradation to input image, and save it to desired output path.

    Args:
        path_image (str): path to input image file
        path_output (str): path to save image to
        intensity_factor (float): factor to reduce intensity of image by
        noise_sigma (int): standard deviation of Gaussian noise added to image
    """

    assert os.path.isfile(path_image), "Input is not a file"

    image = cv2.imread(path_image)
    # image = apply_low_light_degradation(image, intensity_factor, noise_sigma)
    image = apply_low_light_poisson_gaussian(image, noise_sigma=noise_sigma, gamma=gamma, gain=intensity_factor)
    cv2.imwrite(path_output, image)
    print(f"{os.path.split(path_output)[1]} Saved!")

    return image



if __name__ == "__main__":
    dir_root = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/data"
    dir_save = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/img_degraded"

    path_img = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/data/933_cheeseburger/og/20251119_211205.jpg"
    path_img_out = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/augmented/cheeseburger_20251119_211205.jpg"
    path_msrcr = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/augmented/cheeseburger_20251119_211205_msrcr.jpg"
    path_zerodce = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/augmented/cheeseburger_20251119_211205_zerodce.jpg"



    for root, dirs, files in os.walk(dir_root):
        for file in files:
            if file.endswith(".jpg") and root.find("og") != -1:
                img_path = os.path.join(root, file)
                out_path = img_path.replace("data", "img_degraded").replace("og", "")
                os.makedirs(os.path.split(out_path)[0], exist_ok=True)
                image = degrade_lowlight_image(img_path, out_path, intensity_factor=0.5, gamma=2.0)
                # cv2.imwrite(img_path.replace("data", "img_degraded"), image)