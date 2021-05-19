"""
    Helper functions used for evaluation of loss and other metrics
"""
import torch
from utils.config import margin_rank_loss, device, scoring, embed_type, use_embed


def process_text_embedding(text_match, text_diff):
    """
        Process text embedding based on embedding type during training and evaluation

        Args:
            text_match (List[str]/Tensor): For matching caption, list of captions for USE embedding and Tensor for glove/fasttext embeddings
            text_diff (List[str]/Tensor): For non-matching caption, list of captions for USE embedding and Tensor for glove/fasttext embeddings

        Returns:
            text_match (Tensor): Processed text-embedding for matching caption
            text_diff (Tensor): Processed text-embedding for non-matching caption
    """
    if embed_type == 'use':
        text_match = torch.tensor(use_embed(text_match).numpy())
        text_diff = torch.tensor(use_embed(text_diff).numpy())
    text_match = text_match.to(device)
    text_diff = text_diff.to(device)
    return text_match, text_diff


def compute_score(z_img, z_text_match, z_text_diff):
    """
        Computes scores between object(s) and caption-embedding based on provided scoring function

        Args:
            z_img (Tensor): Feature vector of shape (K, N, 300) for bounding box objects, where K and N denotes number of objects and batch size respectively
            z_text_match (Tensor): Feature vector of shape (N, 300)  for matching textual caption
            z_text_diff (Tensor): Feature vector of shape (N, 300) for non-matching textual caption

        Returns:
            score_match (Tensor): Tensor of shape (N, K) holding score of matching caption with each object K in the image
            score_diff (Tensor): Tensor of shape (N, K) holding score of non-matching caption with each object K in the image
    """
    z_img = z_img.permute(1, 0, 2)
    # Compute Scores
    if scoring == 'dot':
        z_text_match = z_text_match.unsqueeze(2)
        z_text_diff = z_text_diff.unsqueeze(2)
        score_match = torch.bmm(z_img, z_text_match).squeeze()
        score_diff = torch.bmm(z_img, z_text_diff).squeeze()
    elif scoring == 'elem':
        z_text_match = z_text_match.unsqueeze(1).repeat(1, 10, 1)
        z_text_diff = z_text_diff.unsqueeze(1).repeat(1, 10, 1)
        score_match = torch.mean(z_img * z_text_match, dim=2)
        score_diff = torch.mean(z_img * z_text_diff, dim=2)
    elif scoring == 'sub':
        z_text_match = z_text_match.unsqueeze(1).repeat(1, 10, 1)
        z_text_diff = z_text_diff.unsqueeze(1).repeat(1, 10, 1)
        score_match = torch.mean(torch.abs(z_img - z_text_match), dim=2)
        score_diff = torch.mean(torch.abs(z_img - z_text_diff), dim=2)
    elif scoring == 'concat':
        z_text_match = z_text_match.unsqueeze(1).repeat(1, 10, 1)
        z_text_diff = z_text_diff.unsqueeze(1).repeat(1, 10, 1)
        score_match = torch.mean(torch.cat((z_img, z_text_match), dim=2), dim=2)
        score_diff = torch.mean(torch.cat((z_img, z_text_diff), dim=2), dim=2)
    return score_match, score_diff


def margin_loss_text_combined(z_img, z_text_match, z_text_diff):
    """
        Computes max-margin loss between objects and text features

        Args:
            z_img (Tensor): Feature vector of shape (K, N, 300) for bounding box objects, where K and N denotes number of objects and batch size respectively
            z_text_match (Tensor): Feature vector of shape (N, 300)  for matching textual caption
            z_text_diff (Tensor): Feature vector of shape (N, 300) for non-matching textual caption

        Returns:
            img_rank_loss (Tensor): Tensor holding scalar value for the loss

    """
    score_match, score_diff = compute_score(z_img, z_text_match, z_text_diff)
    # Rank Images
    sum_match_img = torch.max(score_match, dim=1).values
    sum_diff_img = torch.max(score_diff, dim=1).values
    img_rank_loss = margin_rank_loss(sum_match_img, sum_diff_img, torch.ones(sum_match_img.shape[0]).to(device))
    return img_rank_loss


def get_match_vs_no_match_acc(z_img, z_text_match, z_text_diff):
    """
        Computes how accurately model learns correct matching of object with the caption in terms of accuracy

        Args:
            z_img (Tensor): Feature vector of shape (K, N, 300) for bounding box objects, where K and N denotes number of objects and batch size respectively
            z_text_match (Tensor): Feature vector of shape (N, 300)  for matching textual caption
            z_text_diff (Tensor): Feature vector of shape (N, 300) for non-matching textual caption

        Returns:
            correct (Tensor): Total number of correct predictions
    """

    score_match, score_diff = compute_score(z_img, z_text_match, z_text_diff)
    max_match_img = torch.max(score_match, dim=1).values.unsqueeze(1)   # Computes the score of top object for matching caption
    max_diff_img = torch.max(score_diff, dim=1).values.unsqueeze(1)  # Computes the score of top object for non-matching caption
    combined = torch.cat((max_diff_img, max_match_img), dim=1)
    # If the score for matching caption is higher than non-matching caption, then the prediction is deemed correct
    correct = torch.sum(torch.max(combined, dim=1).indices)
    return correct


def bb_intersection_over_union(boxA, boxB):
    """
        Computes IoU (Intersection over Union for 2 given bounding boxes)

        Args:
            boxA (list): A list of 4 elements holding bounding box coordinates (x1, y1, x2, y2)
            boxB (list): A list of 4 elements holding bounding box coordinates (x1, y1, x2, y2)

        Returns:
            iou (float): Overlap between 2 bounding boxes in terms of overlap
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection area and dividing it by the sum of both boxes
    # intersection area / areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def top_bbox_from_scores(bboxes, scores):
    """
        Returns the top matching bounding box based on scores

        Args:
            bboxes (list): List of bounding boxes for each object
            scores (list): List of scores corresponding to bounding boxes given by bboxes

        Returns:
            matched_bbox: The bounding box with the maximum score
    """
    bbox_scores = [(bbox, score) for bbox, score in zip(bboxes, scores)]
    sorted_bbox_scores = sorted(bbox_scores, key=lambda x: x[1], reverse=True)
    matched_bbox = sorted_bbox_scores[0][0]
    return matched_bbox


def is_bbox_overlap(bbox1, bbox2, iou_overlap_threshold):
    """
        Checks if the two bounding boxes overlap based on certain threshold

        Args:
            bbox1: The coordinates of first bounding box
            bbox2: The coordinates of second bounding box
            iou_overlap_threshold: Threshold value beyond which objects are considered overlapping

        Returns:
            Boolean whether two boxes overlap or not
    """
    iou = bb_intersection_over_union(boxA=bbox1, boxB=bbox2)
    if iou >= iou_overlap_threshold:
        return True
    return False
