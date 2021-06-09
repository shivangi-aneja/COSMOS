import os.path
import torch
import itertools as it
from torch.utils.data import Dataset
from utils.dataset_utils import modify_caption_replace_entities
from utils.common_utils import read_json_data
from utils.config import num_boxes, embed_type, DATA_DIR
from utils.custom_transforms.data_aug import *


class CaptionInContext(Dataset):
    """Custom dataset class for Out-of-Context Detection"""

    def __init__(self, metadata_file, mode, transforms, text_field=None):
        """
            Initializes the dataset object
            Args:
                metadata_file (string): Path to the json file with annotations.
                mode (string): train, val, test.
                transforms (callable): Transform to be applied on a sample.
                text_field (torchtext.Field): Vocab object for applying text transformations (used for Glove and Fasttext embeddings only)
            Returns:
                None
        """
        self.data = read_json_data(metadata_file)
        self.mode = mode
        self.transforms = transforms
        self.text_field = text_field
        self.flip_rotate_transform = Sequence(
            [RandomHorizontalFlip(0.8), RandomScale(0.2, diff=True), RandomRotate(10)])

    def __getitem__(self, index):
        """
            Returns sample corresponding to the index `index`
        """
        img_data = self.data[index]
        img_path = os.path.join(DATA_DIR, img_data['img_local_path'])

        # Predicted Mask R-CNN bounding boxes, we use top 10 boxes for our experiments
        bboxes = img_data['maskrcnn_bboxes'][:10]
        # We do not take into account the bounding box classes,
        # but they need to passed later for extracting bbox features
        bbox_class = [-1] * len(bboxes)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # During training, apply random flip, scale and rotate transformation
        if self.mode == 'train':
            try:
                img_aug, bboxes_aug = self.flip_rotate_transform(img, np.array(bboxes))
                bboxes_aug = bboxes_aug.tolist()
                bboxes = list(it.islice(it.cycle(bboxes_aug), num_boxes - 1))
                img = img_aug
            except:
                pass

        # Consider entire image as additional object for global context
        img = self.transforms(img).permute(1, 2, 0)
        img_shape = img.shape[:2]
        bboxes.append([0, 0, img_shape[1], img_shape[0]])
        bbox_class.append(-1)

        if self.mode == 'test':
            idx1 = random.randint(0, 1)
            cap_key1 = 'caption1' if idx1 == 0 else 'caption2'
            caption1 = img_data[cap_key1]
            caption1 = modify_caption_replace_entities(caption1)

            while True:
                idx2 = random.randint(0, 1)
                cap_key2 = 'caption1' if idx2 == 0 else 'caption2'
                tgt_index = random.randint(0, len(self.data) - 1)
                caption2 = self.data[tgt_index][cap_key2]
                caption2 = modify_caption_replace_entities(caption2)
                if caption1 != caption2:
                    break
        else:
            src_captions = img_data['articles']
            caption1 = src_captions[random.randint(0, len(src_captions) - 1)]['caption_modified']

            while True:
                tgt_index = random.randint(0, len(self.data) - 1)
                tgt_captions = self.data[tgt_index]['articles']
                caption2 = tgt_captions[random.randint(0, len(tgt_captions) - 1)]['caption_modified']
                if caption1 != caption2:
                    break
        # Compute text-embeddings for Glove and Fasttext embeddings
        if embed_type != 'use':
            text_match = self.text_field.preprocess(caption1)
            text_match = torch.stack([self.text_field.vocab.vectors[self.text_field.vocab.stoi[x]] for x in text_match])

            text_diff = self.text_field.preprocess(caption2)
            text_diff = torch.stack([self.text_field.vocab.vectors[self.text_field.vocab.stoi[x]] for x in text_diff])
        else:
            # For USE embeddings, embeddings will be evaluated in trainer script
            text_match = caption1
            text_diff = caption2

        return img, text_match, text_diff, len(text_match), len(text_diff), bboxes, bbox_class

    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.data)
