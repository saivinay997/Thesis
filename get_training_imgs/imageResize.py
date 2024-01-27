import torch
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF
from PIL import Image
import os
from scipy.io import loadmat
import sys
# sys.path.append('..')
from KernelGANimgResize.imresize import imresize
import utils
import numpy as np

tdsr_hr_dir = r"C:\SaiVinay\Thesis\Code\get_training_imgs\results\Test_1\imgsResizedHR"
tdsr_lr_dir = r"C:\SaiVinay\Thesis\Code\get_training_imgs\results\Test_1\imgsResizedLR"


def get_img_resize(input_source_dir, opt_kernel_path, upscale_factor=4, cleanup_factor=1, tdsr_hr_dir = tdsr_hr_dir, tdsr_lr_dir=tdsr_lr_dir):
    """This function is to resize the images in the input_source_dir using the kernel from opt_kernel_path to the upscale_factor"""
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    print(f"Number of images in the source directory: {len(source_files)}")
    with torch.no_grad():
        for file in tqdm(source_files, desc="Resizing images"):
            # load HR image
            input_img = Image.open(file)
            input_img = TF.to_tensor(input_img)
            
            print("input_img.size(): ", input_img.size())
            resize2_img = utils.imresize(input_img, 1.0 / cleanup_factor, True)
            _, w, h = resize2_img.size()
            
            print("resize2_img.size(): ", resize2_img.size())
            w = w - w % upscale_factor
            h = h - h % upscale_factor
            resize2_cut_img = resize2_img[:, :w, :h]
            
            print("resize2_cut_img.size(): ", resize2_cut_img.size())


            try:
                file_name = os.path.basename(file)
                file_id = file_name.split('.')[0]
                kernel_path = os.path.join(opt_kernel_path,file_id+'\\'+file_id+'_kernel_x4.mat')
                mat = loadmat(kernel_path)
                # print(kernel_path)

                path = os.path.join(tdsr_hr_dir, os.path.basename(file))
                if not os.path.exists(path):
                    print(f'create_HrLr kernel_path:{kernel_path}')
                    HR_img = TF.to_pil_image(input_img)
                    HR_img.save(path, 'PNG')

                    k = np.array([mat['Kernel']]).squeeze()
                    # print(type(k), k.shape, k.dtype)
                    resize_img = imresize(np.array(HR_img), scale_factor=1.0 / upscale_factor, kernel=k)

                    path = os.path.join(tdsr_lr_dir, os.path.basename(file))
                    TF.to_pil_image(resize_img).save(path, 'PNG')
                else:
                    print(f'skip kernel_path:{kernel_path}')

            except Exception as e:
                print(e)