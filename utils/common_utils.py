"""
    Utility file consisting of common functions and variables used during training and evaluation
"""

import json
import os
import cv2
from utils.config import TARGET_DIR


def read_json_data(file_name):
    """
        Utility function to read data from json file

        Args:
            file_name (str): Path to json file to be read

        Returns:
            article_list (List<dict>): List of dict that contains metadata for each item
    """
    with open(file_name) as f:
        article_list = [json.loads(line) for line in f]
        return article_list


def draw_bboxes(file_name):
    """
    Utility functions to visualize the bounding boxes

    Args:
        file_name (str): Path for the file to be read

    Returns:
         None
    """
    box_color = (0, 255, 0)  # Green
    line_thickness = 5
    data_list = read_json_data(file_name)
    for data in data_list:
        img_path = data["img_local_path"]
        img = cv2.imread(img_path)
        bboxes = data["maskrcnn_bboxes"]

        for i, bbox in enumerate(bboxes):
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color,
                                line_thickness)
        cv2.imwrite(os.path.join(TARGET_DIR, data["img_local_path"]), img)
