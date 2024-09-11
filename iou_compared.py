import os

root_path = '/home/shannon/local/Source/Python/bm_study/unet/'
run_name = 'run_unet_texture_01'

results_path = os.path.join(root_path, run_name, 'predictions_ranked')

results_files = [f for f in os.listdir(results_path) if f.endswith('.png') and 'original' in f]

iou_textures = []
for filename in results_files:
    iou_textures.append(int(filename.split('_')[0]))


run_name = 'run_unet_images_01'
results_path = os.path.join(root_path, run_name, 'predictions_ranked')

results_files = [f for f in os.listdir(results_path) if f.endswith('.png') and 'original' in f]

iou_images = []
for filename in results_files:
    iou_images.append(int(filename.split('_')[0]))

root_path = '/home/shannon/local/Source/Python/bm_study/yolo/'
run_name = 'run_yolo_01'
results_path = os.path.join(root_path, run_name, 'test/images/results')

results_files = [f for f in os.listdir(results_path) if f.endswith('.png') and 'result' in f]

iou_yolo = []
for filename in results_files:
    iou_yolo.append(int(filename.split('_')[0]))

print(iou_yolo)


import matplotlib.pyplot as plt

# Combine all IOU values into a single list
all_iou = [iou_yolo, iou_images, iou_textures]

# Plot the histogram
plt.hist(all_iou, bins=10, label=['IoU YOLO', 'IoU U-Net Images', 'IoU U-Net Textures'])
plt.xlabel('IoU')
plt.ylabel('Frequency')
plt.legend()

# Save the plot as 'iou_compared.png'
plt.savefig('/home/shannon/local/Source/Python/bm_study/iou_compared.png')

# Display the histogram
plt.show()