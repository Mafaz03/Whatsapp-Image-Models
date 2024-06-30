import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import re

def extract_color(src, hsv_color1, hsv_color2, blur=(20, 20), threshold=None, show = False):
    img = cv.imread(src) if type(src) is str else src
    if blur != (0,0): img = cv.blur(img, blur)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, hsv_color1, hsv_color2)
    if threshold is not None:
        _, mask = cv.threshold(mask, threshold[0], threshold[1], cv.THRESH_BINARY)
    if show:
        plt.imshow(mask, cmap='gray')
        plt.axis('off')  
        plt.show()
    
    return mask

def get_mask_cord(mask):
    non_black_pixels = np.argwhere(mask > 0)
    top_left = non_black_pixels.min(axis=0)[::-1]
    bottom_right = non_black_pixels.max(axis=0)[::-1]
    bounding_box = (*top_left, *bottom_right)
    return bounding_box

def draw_bounding_box(mask, threshold = 230, kernel_size = 5, strict = False, show = False):
    if strict:
        bbox = get_mask_cord(mask)
        top_left, bottom_right = bbox[:2], bbox[2:]

        mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        cv.rectangle(mask_rgb, pt1=top_left, pt2=bottom_right, color=(250, 0, 0), thickness=20)
        if show:
            plt.imshow(mask_rgb)
            plt.axis('off')
            plt.show()
        return bbox
        
    _, thresh = cv.threshold(mask, threshold, 255, cv.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    
    # Find contours in the thresholded image
    contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour which we assume is the document
    largest_contour = max(contours, key=cv.contourArea)
    x1, y1, w, h = cv.boundingRect(largest_contour)
    x2, y2 = x1+w, y1+h
    image_with_box = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    cv.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if show:
        plt.imshow(image_with_box, cmap='gray')
        plt.axis('off')
        plt.show()

    return x1, y1, x2, y2

def WhatsappSegment(src, alpha = 0.1, kernel_size=5, threshold = 240, show = False):
    im = plt.imread(src) if type(src) is str else src
    overlay = im.copy()

    history_mask = extract_color(im, (0,0,30), (0,0,30), blur=(10,10), threshold=(threshold,255))
    history_bbox = draw_bounding_box(history_mask, kernel_size=kernel_size, show=False)

    chat_mask = extract_color(im, (0,0,0), (0,0,0), threshold=(threshold, 255))
    chat_bbox = draw_bounding_box(chat_mask, strict=True, show=False)

    colors = [
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (0, 255, 255), # Cyan
    (255, 0, 255), # Magenta
    (255, 255, 0),  # Yellow
    (255, 0, 0),   # Red
    ]

    menu_bbox = [0, 0, history_bbox[0], history_bbox[-1]]
    text_box = [chat_bbox[0], chat_bbox[-1], im.shape[1] , im.shape[0]]
    search_bbox = [history_bbox[0], chat_bbox[1], history_bbox[2], history_bbox[1]] if history_bbox[1] > chat_bbox[-1] else [history_bbox[0], chat_bbox[1], history_bbox[2], history_bbox[0]]
    name_bbox = [chat_bbox[0], 0, chat_bbox[2], chat_bbox[1]]
    if history_bbox[1] < chat_bbox[1]: 
        history_bbox = list(history_bbox)
        history_bbox[1] = chat_bbox[1]
        history_bbox = tuple(history_bbox)
    
    if show:
        for i, bbox in enumerate([chat_bbox, history_bbox, menu_bbox, text_box, search_bbox, name_bbox]):
            im = cv.rectangle(img=im, pt1=bbox[:2], pt2=bbox[2:], color=colors[i], thickness=10)

            cv.rectangle(img=overlay, pt1=bbox[:2], pt2=bbox[2:], color=colors[i], thickness=-1)
            image_new = cv.addWeighted(overlay, alpha, im, 1 - alpha, 0)

        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        image_new = cv.cvtColor(image_new, cv.COLOR_BGR2RGB)

        plt.figure(figsize=(10,10))
        plt.imshow(image_new)
        plt.axis("off")
        plt.show()

    return {"chat_bbox": chat_bbox, "history_bbox": history_bbox, "menu_bbox": menu_bbox, 
            "text_box": text_box, "search_bbox": search_bbox, "name_bbox": name_bbox}

def remove_circle(src, ismask=False , bbox=None, show=False, dp=0.01, 
                  minDist=30, param1=50, param2=100, minRadius=15, maxRadius=300, 
                  circle_color=(255,255,255), thickness=-1, col1=(0,0,0), col2 = (0,0,0)) -> Tuple[np.array, np.array]:
    circles_list = []
    chat_mask = plt.imread(src) if type(src) is str else src
    if bbox: chat_image = chat_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else: chat_image = chat_mask
    if not ismask: chat_mask = extract_color(chat_image, col1, col2, blur=(1,1))
    blurred = cv.GaussianBlur(chat_mask, (9, 9), 2)
    circles = cv.HoughCircles(blurred, 
                           cv.HOUGH_GRADIENT, 
                           dp=dp, 
                           minDist=minDist, 
                           param1=param1, 
                           param2=param2, 
                           minRadius=minRadius, 
                           maxRadius=maxRadius)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv.circle(chat_mask, (x, y), r+10, circle_color, thickness=thickness)
        print(f"{len(circles)} found!!")
    else:
        print("No circles were found")
    if show:
        plt.imshow(chat_mask, cmap="gray")
        plt.axis('off')
        plt.show()
    return chat_mask, circles

def chat_message_bbox(mask, blur=10, kenel=8, black_thresh=100, min_area=500, show=False):
    chat_message_bbox = []
    blur_image = cv.blur(mask,(blur,blur))
    black_thresh = black_thresh
    _, black_mask = cv.threshold(blur_image, black_thresh, 255, cv.THRESH_BINARY_INV)
    kernel = np.ones((kenel, kenel), np.uint8)
    black_mask = cv.morphologyEx(black_mask, cv.MORPH_CLOSE, kernel)
    black_contours, _ = cv.findContours(black_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    for contour in black_contours:
        if cv.contourArea(contour) > min_area:
            x1, y1, w, h = cv.boundingRect(contour)
            x2, y2 = x1 + w, y1 + h
            chat_message_bbox.append((x1, y1, x2, y2,))
            cv.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 5)
     
    if show:
        plt.imshow(image_bgr, cmap="gray")
        plt.axis("off")
        plt.show()
    return chat_message_bbox

def whosaid(message_bbox):
    x1x2 = [[i[0],i[2]] for i in message_bbox]
    values = [min([i[0] for i in x1x2]), max([i[1] for i in x1x2])]
    labels = np.argmin(np.abs(np.array(x1x2) - values),1)
    return labels
