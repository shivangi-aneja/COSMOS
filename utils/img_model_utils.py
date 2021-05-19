import torch
from detectron2.structures import (Boxes, BoxMode, Instances)


# Partially borrowed from Detectron2 source code
def annotations_to_instances(bboxes, bbox_classes, image_size):
    """
        Create an :class:`Instances` object used by the models, from instance annotations in the dataset dict.

        Args:
            bboxes (ndarray): numpy array of shape (K, 4) where K denotes no. of objects in the image and 4 bounding box coordinate for each object
            bbox_classes (ndarray): numpy array of shape (K,) holding dummy values for class label where K denotes no. of objects in the image
            image_size (tuple): height, width

        Returns:
            Instances: It will contain fields "bboxes", "classes", This is the format that builtin Detectron models expect.
    """
    boxes = [BoxMode.convert(obj, BoxMode.XYXY_ABS, BoxMode.XYXY_ABS) for obj in bboxes]
    target = Instances(image_size)
    target.bboxes = Boxes(boxes)
    classes = [int(obj) for obj in bbox_classes]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.classes = classes
    return target
