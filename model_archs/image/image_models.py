"""
    Class file that enlists models for extracting features from image
"""
import torch
from torch import nn
from utils.img_model_utils import annotations_to_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


class MaskRCNNExtractor(nn.Module):
    def __init__(self):
        """
            Initializes the Mask-RCNN predictor that would be used to extract bounding box features

            Returns:
                None
        """

        super(MaskRCNNExtractor, self).__init__()
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set RPN threshold for this model
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
        self.maskrcnn_predictor = DefaultPredictor(cfg)

    def forward(self, imgs, bboxes, bbox_classes):
        """
            Function to compute forward pass of the network

            Args:
                imgs (List[Tensor]): list of length N of images (C X W X H), where N denotes minibatch size, C, H, W denotes image channels, width and height
                bboxes (List[Tensor]): bounding boxes corresponding to the images.
                bbox_classes (List[Tensor]): Dummy parameter required by Mask-RCNN, not used otherwise

            Returns:
                bbox_feats (Tensor): Bounding box features. Tensor of shape (N X K X 2048 X 7 X 7), where N is the batch size, K denotes number of objects and features map of shape 2048 X 7 x 7 is extracted for each object
        """
        img_shapes = [img.cpu().numpy().shape[:2] for img in imgs]
        # Change to tensor format for input to model
        targets = [annotations_to_instances(bbox.cpu().numpy(), bbox_class.cpu().numpy(), img_shape) for
                   bbox, bbox_class, img_shape in
                   zip(bboxes, bbox_classes, img_shapes)]
        bbox_feats = torch.stack(
            [self.maskrcnn_predictor(img.cpu().numpy(), target) for img, target in zip(imgs, targets)])
        return bbox_feats


class ProcessMaskRCNNFeats(nn.Module):
    def __init__(self):
        """
            Initializes the model to process bounding box features extracted by MaskRCNNExtractor

            Returns:
                None
        """
        super(ProcessMaskRCNNFeats, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 300)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
            Function to compute forward pass of the network

        Args:
            x (Tensor): Bounding box features extracted from Mask R-CNN. Tensor of shape (N X 2048 X 7 X 7), where N is the batch size and features map of shape 2048 X 7 x 7 for each object

        Returns:
            x (Tensor): Processed bounding box features. Tensor of shape (N X 300)
            """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc2(self.relu(self.fc1(x)))
        return x
