
import os
from . import zero_dce_model
import torch
import torchvision
import numpy as np
import skimage
import matplotlib.pyplot as plt

from PIL import Image


def save_image(image, path_output):
	"""
	Saves input image to file.
	"""
	torchvision.utils.save_image(image, path_output)
	print(f"{os.path.split(path_output)[1]} Saved!")


def infer_image(path_image: str, path_ckpt: str = "") -> np.ndarray:
    """
    Uses zero-DCE++ model to enhance input image.
    Image can be saved to file using save_image function.
    
    Args:
        path_image: path to input image
        path_ckpt: path to Epoch99.pth
    Returns:
        np.ndarray: numpy array of enhanced image
    """
    scale_factor = 12

    if len(path_ckpt) == 0:
        script_path = os.path.abspath(__file__)
        path_ckpt = os.path.join(os.path.split(script_path)[0], "Epoch99.pth")
    
    with torch.no_grad():

        image = Image.open(path_image)
        image = torch.from_numpy(np.asarray(image)/255.0).float()
        
        h, w = image.shape[0] // scale_factor * scale_factor, image.shape[1] // scale_factor * scale_factor
        
        image = image[0:h, 0:w, :].permute(2, 0, 1)
        image = image.cuda().unsqueeze(0)
        
        DCE_net = zero_dce_model.enhance_net_nopool(scale_factor).cuda()
        DCE_net.load_state_dict(torch.load(path_ckpt))
        
        enhanced_image, param_map = DCE_net(image)
    
    return enhanced_image.detach().cpu().numpy().squeeze(0)


if __name__ == '__main__':
    path_ckpt = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/Zero-DCE_extension/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"
    path_image = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/data/917_comic_book/low_light/20251119_160748.jpg"
    path_save = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/data/917_comic_book/low_light_zerodce/20251119_160748.jpg"

    image_original = Image.open("/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/data/917_comic_book/og/20251119_160748.jpg")
    image = skimage.io.imread(path_image)
    enhanced_image = infer_image(path_image, path_ckpt)
    # save_image(enhanced_image, path_save)

    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(image_original.transpose((1, 0, 2)))
    axs[1].imshow(image)
    axs[2].imshow(enhanced_image.transpose((1, 2, 0)))
    plt.show()

		

