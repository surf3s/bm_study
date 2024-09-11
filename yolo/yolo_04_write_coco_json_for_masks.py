'''
Code to write the masks to COCO format
Based on code here https://github.com/brunobelloni/binary-to-coco-json-converter
'''

import glob
import json
import os
import cv2


MASK_EXT = 'png'
ORIGINAL_EXT = 'png'
image_id = 0


def find_contours(sub_mask):
    gray = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


def create_category_annotation(category_dict):
    category_list = []
    for key, value in category_dict.items():
        category = {"id": value, "name": key, "supercategory": key}
        category_list.append(category)
    return category_list


def create_image_annotation(file_name, width, height):
    global image_id
    image_id += 1
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
    }


def create_annotation_format(contour, image_id_, category_id, annotation_id):
    return {
        "iscrowd": 0,
        "id": annotation_id,
        "image_id": image_id_,
        "category_id": category_id,
        "bbox": cv2.boundingRect(contour),
        "area": cv2.contourArea(contour),
        "segmentation": [contour.flatten().tolist()],
    }


def get_coco_json_format():
    return {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}],
    }


# Get "images" and "annotations" info
def images_annotations_info(maskpath):
    annotation_id = 0
    annotations = []
    images = []

    for category in category_ids.keys():
        for mask_image in glob.glob(os.path.join(maskpath, category, f'*.{MASK_EXT}')):
            original_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{ORIGINAL_EXT}'
            mask_image_open = cv2.imread(mask_image)
            height, width, c = mask_image_open.shape

            if original_file_name not in map(lambda img: img['file_name'], images):
                image = create_image_annotation(file_name=original_file_name, width=width, height=height)
                images.append(image)
            else:
                image = [element for element in images if element['file_name'] == original_file_name][0]

            contours = find_contours(mask_image_open)

            for contour in contours:
                annotation = create_annotation_format(contour, image['id'], category_ids[category], annotation_id)
                if annotation['area'] > 0:
                    annotations.append(annotation)
                    annotation_id += 1

    return images, annotations, annotation_id

# Label ids of the dataset
category_ids = {
    "resorption": 1,
}

coco_format = get_coco_json_format()  # Get the standard COCO JSON format

run_name = "run_yolo_01"

for keyword in ["valid", "test", "train"]:
    mask_path = run_name + f"/{keyword}/labels/"

    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

    with open(run_name + f"/{keyword}.json", "w") as outfile:
        json.dump(coco_format, outfile, sort_keys=True, indent=4)

    print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))