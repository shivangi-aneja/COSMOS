""" This file list architecture to combine image-objects and text-embeddings"""

import torch
from torch import nn
from model_archs.image.image_models import MaskRCNNExtractor, ProcessMaskRCNNFeats
from model_archs.text.text_models import ToyRNNLSTM, ToyText
from utils.config import device, num_boxes


class CombinedModelMaskRCNN(nn.Module):
    def __init__(self, hidden_size, use=True, embedding_length=None):
        """
            Creates an instance for the model that perform image-text matching task
            Args:
                hidden_size (int): dimensionality of the latent space
                use (bool): whether to use Universal Sentence Encoder (USE) to embed text
                embedding_length: Size of word embedding vector (used only for Glove and Fasttext)

            Returns:
                None
        """
        super(CombinedModelMaskRCNN, self).__init__()
        self.maskrcnn_extractor = MaskRCNNExtractor()
        self.img_model = ProcessMaskRCNNFeats()
        self.use = use
        if self.use:
            self.text_model = ToyText(hidden_size)
        else:
            self.text_model = ToyRNNLSTM(hidden_size, embedding_length)

    def forward(self, img, text_match, text_diff, batch_size, seq_len_match, seq_len_diff, bboxes, bbox_classes):
        """
            Computes the forward pass of the network

            Args:
                img (List): list of length N of images (C X W X H), where N denotes minibatch size, C, H, W denotes image channels, width and height
                text_match (Tensor): Text embeddings of matching caption
                text_diff (Tensor): Text embeddings of non-matching caption
                batch_size (int): Mini-batch size
                seq_len_match (list): Sequence(word) length for matching captions in the minibatch (used only for Glove and Fasttext)
                seq_len_diff (list): Sequence(word) length for non-matching captions in the minibatch (used only for Glove and Fasttext)
                bboxes (List): bounding boxes corresponding to the images.
                bbox_classes (List): Dummy parameter required by Mask-RCNN, not used otherwise

            Returns:
                z_img (Tensor): Processed bounding box features. Tensor of shape (N X K X 300) where N is the batch size and K is the number of objects
                z_text_match (Tensor): Processed feature vector of shape (N X 300) for matching caption where N is the batch size
                z_text_diff (Tensor): Processed feature vector of shape (N X 300) for non-matching caption where N is the batch size

        """
        img = self.maskrcnn_extractor(img, bboxes, bbox_classes)
        z_img = torch.stack([self.img_model(img[:, i, :, :, :].to(device)) for i in range(num_boxes)])
        if self.use:
            z_text_match = self.text_model(text_match)
            z_text_diff = self.text_model(text_diff)
        else:
            z_text_match = self.text_model(text_match, batch_size, seq_len_match)
            z_text_diff = self.text_model(text_diff, batch_size, seq_len_diff)
        return z_img, z_text_match, z_text_diff
