"""
    Utility file with helper functions for data loading
"""

import torch
from utils.config import embed_type
import spacy
nlp = spacy.load("en_core_web_sm")


def modify_caption_replace_entities(caption_text):
    """
        Utility function to replace named entities in the caption with their corresponding hypernyms

        Args:
            caption_text (str): Original caption with named entities

        Returns:
            caption_modified (str): Modified caption after replacing named entities
    """
    doc = nlp(caption_text)
    caption_modified = caption_text
    caption_entity_list = []
    for ent in doc.ents:
        caption_entity_list.append((ent.text, ent.label_))
        caption_modified = caption_modified.replace(ent.text, ent.label_, 1)
    return caption_modified


def pad_tensor(vec, pad, dim):
    """
        Pads a tensor with zeros according to arguments given

        Args:
            vec (Tensor): Tensor to pad
            pad (int): The total tensor size with pad
            dim (int): Dimension to pad

        Returns:
            padded_tensor (Tensor): A new tensor padded to 'pad' in dimension 'dim'

    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    padded_tensor = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    return padded_tensor


class PadCollate:
    def __init__(self, img_dim=0, embed_dim1=1, embed_dim2=2):
        """
        Args:
            img_dim (int): dimension for the image bounding boxes
            embed_dim1 (int): dimension for the matching caption
            embed_dim2 (int): dimension for the non-matching caption
        """
        self.img_dim = img_dim
        self.embed_dim1 = embed_dim1
        self.embed_dim2 = embed_dim2

    def pad_collate(self, batch):
        """
            A variant of collate_fn that pads according to the longest sequence in a batch of sequences and forms the minibatch

            Args:
                batch (list): list of (img, text_match, text_diff, len(text_match), len(text_diff), bboxes, bbox_class)

            Returns:
                Tensors/List of the image and text features for convenient processing on GPU
        """

        if embed_type == 'use':
            t1 = list(map(lambda x: x[self.embed_dim1], batch))
            t2 = list(map(lambda x: x[self.embed_dim2], batch))
            # seq_len not needed for use embeddings
            seq_len1 = 0
            seq_len2 = 0
        else:
            max_len1 = max(map(lambda x: x[self.embed_dim1].shape[0], batch))
            max_len2 = max(map(lambda x: x[self.embed_dim2].shape[0], batch))
            # Add zero padding to smaller sentences in the minibatch to ensure tensor formation
            embed_batch1 = list(map(lambda t: pad_tensor(t[self.embed_dim1], pad=max_len1, dim=0), batch))
            embed_batch2 = list(map(lambda t: pad_tensor(t[self.embed_dim2], pad=max_len2, dim=0), batch))
            t1 = torch.stack(embed_batch1, dim=0)
            t2 = torch.stack(embed_batch2, dim=0)
            seq_len1 = torch.LongTensor(list(map(lambda x: x[self.embed_dim1].shape[0], batch)))
            seq_len2 = torch.LongTensor(list(map(lambda x: x[self.embed_dim2].shape[0], batch)))

        # stack all
        xs = list(map(lambda t: t[self.img_dim].clone().detach(), batch))
        bboxes = list(map(lambda t: torch.tensor(t[-2]), batch))
        bbox_classes = list(map(lambda t: torch.LongTensor(t[-1]), batch))
        return xs, t1, t2, seq_len1, seq_len2, bboxes, bbox_classes

    def __call__(self, batch):
        return self.pad_collate(batch)

