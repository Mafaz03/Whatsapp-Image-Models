import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bbox_pred(result):
  data = result[0].orig_img
  fig, ax = plt.subplots()
  ax.imshow(data)
  plt.axis("off")
  for idx, box in enumerate(result[0].boxes.xywh):
    x_center, y_center, width, height = box.cpu()
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor="y", facecolor="y", alpha=0.5)
    ax.add_patch(rect)
  print(f"Shape: {data.shape}")