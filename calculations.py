from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from PIL import ImageOps
from plot_helper import *


def resize_and_center_image(target_shape, image_path = None, image_np = None):
    target_width, target_height = target_shape
    if image_path is not None: img = Image.open(image_path)
    elif image_np is not None : img = Image.fromarray(image_np.astype('uint8'), 'RGB')
    else: raise ValueError("Either image_path or image_np must be provided")
    # 3 channel max
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = ImageOps.contain(img, (target_width,target_height))
    img.thumbnail((target_width, target_height), Image.LANCZOS)
    canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    x_offset = (target_width - img.width) // 2
    y_offset = (target_height - img.height) // 2
    canvas.paste(img, (x_offset, y_offset))
    image_array = np.array(canvas)
    return image_array


def transform_to_tensor(image, show = False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(image)
    print(f"Shape of the RGB image tensor: {img_tensor.shape}")
    if show:
        plt.imshow(img_tensor.permute(1, 2, 0))
        plt.axis("off")
        plt.show()

    return img_tensor


def predict_and_show(model, image_path = None, image = None, target_shape=(1920,1080)):
    # if shape%32 != 0:
    #     shape = closestNumber(shape, 32)
    #     print(f"Shape converted to {shape}")
    #     print(f"!! Please used {shape} here after")
    if image_path is not None:
        image = resize_and_center_image(target_shape=target_shape, image_path = image_path)
    elif image is not None: 
        image = resize_and_center_image(target_shape=target_shape, image_np = image)
    plot, bbox = plot_bbox_pred(array_image = image, model=model)
    return plot, bbox

def closestNumber(n, m) :
    q = int(n / m)
    n1 = m * q
    if((n * m) > 0) : n2 = (m * (q + 1)) 
    else : n2 = (m * (q - 1))
    if (abs(n - n1) < abs(n - n2)): return n1
    return n2


def resize_bbox(bbox, current_shape, target_shape):
    current_width, current_height = current_shape
    new_width, new_height = target_shape
    scale_factor_width = new_width / current_width
    scale_factor_height = new_height / current_height
    
    x_min, y_min, x_max, y_max = bbox
    new_x_min = int(scale_factor_width * x_min)
    new_y_min = int(scale_factor_height * y_min)
    new_x_max = int(scale_factor_width * x_max)
    new_y_max = int(scale_factor_height * y_max)
    
    return (new_x_min, new_y_min, new_x_max, new_y_max)


def crop_resizebbox(image_path, current_shape, target_shape, boxes_xyxy, margin_shape:(int,int)):
    current_width, current_height = current_shape
    target_width, target_height = target_shape
    new_x_min, new_y_min, new_x_max, new_y_max = resize_bbox(boxes_xyxy, current_shape, target_shape)
    new_x_min = max(new_x_min - margin_shape[0], 0)
    new_y_min = max(new_y_min - margin_shape[1], 0)
    new_x_max = min(new_x_max + margin_shape[0], target_width)
    new_y_max = min(new_y_max + margin_shape[1], target_height)

    image = resize_and_center_image(image_path=image_path, target_shape=target_shape)
    image_cropped = image[new_y_min: new_y_max, new_x_min: new_x_max]
    print(f"Shape: {image_cropped.shape}")
    return image_cropped