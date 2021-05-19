""" Basic configuration and settings for training the model"""

import torch
from torch import nn
import tensorflow_hub as hub
import torchvision.transforms as transforms

# Data Directories
BASE_DIR = '/home/shivangi/Desktop/Data/Projects/cosmos/'
DATA_DIR = '/home/shivangi/Desktop/Data/Projects/cosmos/data/'
TARGET_DIR = "/home/shivangi/Desktop/Data/Projects/cosmos/viz/"

# Word Embeddings
embedding_length = 300
embed_type = 'use'  # glove, fasttext, use
use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Hyperparameters
patience = 10
batch_size = 64
epochs = 500
# Optimizers
lr = 1e-3
img_lr = 1e-3
text_lr = 1e-3

# Losses
cse_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')
margin_rank_loss = nn.MarginRankingLoss(margin=1)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Pre-processing
img_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
img_transform_train = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(hue=.2, saturation=.2), transforms.ToTensor(),  transforms.Normalize([0.5] * 3, [0.5] * 3)])

# Image Partitioning
num_boxes = 11  # Number of bounding boxes used in experiments, one additional box for entire image (global context)
retrieve_gt_proposal_features = True    # Flag that controls whether to retrieve bounding box features from Mask RCNN backbone or not
scoring = 'dot'  # Scoring function to combine image and text embeddings

iou_overlap_threshold = 0.5
textual_sim_threshold = 0.5
