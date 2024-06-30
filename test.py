import cv2 as cv
from PIL import Image
import random

# Load and resize the image
image = Image.open("/Users/mohamedmafaz/Whatsapp-Image-Models/360_F_124963716_Rb81mdhUZrYgvnvuhIvjKQReZyORMLxe.jpg")
image_re = image.resize((100, 40))
digits = ['g','3','5','7','#','@']

shape = image_re.size

for i in range(shape[1]):
    line = ''
    for j in range(shape[0]):
        r, g, b = image_re.getpixel((j, i))
        rgb_color_code = f"\033[38;2;{r};{g};{b}m"
        element = random.choice(digits)
        line += f"{rgb_color_code}{element}\033[0m"
    print(line)

go