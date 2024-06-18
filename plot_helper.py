import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

def plot_bbox_pred(image_path = None, PIL_image = None, array_image = None, model = None, output_path = None, axis = "off"):

    if image_path is not None:
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            PIL_image = img.convert('RGB')
    elif array_image is not None:
        PIL_image = Image.fromarray(array_image, 'RGB')
    if type(PIL_image) == Image.Image and PIL_image.mode == 'RGBA':
        img = img.convert('RGB')
    else: img = PIL_image
    if model is None: raise ValueError("Enter Model")
    prediction = model.predict(img)[0]
    plot = prediction.plot()
    plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    bbox = prediction.boxes

    plt.imshow(plot)
    plt.axis(axis)
    if output_path: plt.imsave(output_path, plot)

    return plot, bbox