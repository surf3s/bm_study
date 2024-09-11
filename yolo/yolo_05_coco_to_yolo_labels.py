# Convert the COCO masks to YOLO masks
from ultralytics.data.converter import convert_coco
import os
import shutil

run_name = 'run_yolo_01'

convert_coco(run_name, os.path.join(run_name, 'labels'), use_segments=True)

# I am not how to tell this converter to use my folder structure, so now move these files to where I need them,
# and clean up after.

source_dir = os.path.join(run_name, 'labels', 'labels', 'train')
destination_dir = os.path.join(run_name, 'train', 'labels')
shutil.move(source_dir, destination_dir)

source_dir = os.path.join(run_name, 'labels', 'labels', 'valid')
destination_dir = os.path.join(run_name, 'valid', 'labels')
shutil.move(source_dir, destination_dir)

source_dir = os.path.join(run_name, 'labels', 'labels', 'test')
destination_dir = os.path.join(run_name, 'test', 'labels')
shutil.move(source_dir, destination_dir)

# now delete the folder run_name+labels
shutil.rmtree(os.path.join(run_name, 'labels'))

