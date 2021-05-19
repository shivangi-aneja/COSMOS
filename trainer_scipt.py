""" Script file to train the model """

import os
import numpy as np
import argparse
from tqdm import tqdm
import torch.optim as optim
from utils.logging.tf_logger import Logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_archs.models import CombinedModelMaskRCNN
from utils.text_utils import get_text_metadata
from torch.utils.data import DataLoader
from utils.dataset_utils import PadCollate
from utils.eval_utils import get_match_vs_no_match_acc, margin_loss_text_combined, process_text_embedding
from utils.config import *
from utils.dataset import CaptionInContext


# Word Embeddings
text_field, word_embeddings, vocab_size = get_text_metadata()

# DataLoaders
train_dataset = CaptionInContext(metadata_file=os.path.join(DATA_DIR, 'annotations', 'train_data.json'),
                                 transforms=img_transform_train, mode='train', text_field=text_field)

val_dataset = CaptionInContext(metadata_file=os.path.join(DATA_DIR, 'annotations', 'val_data.json'),
                               transforms=img_transform, mode='val', text_field=text_field)

test_dataset = CaptionInContext(metadata_file=os.path.join(DATA_DIR, 'annotations', 'test_data.json'),
                               transforms=img_transform, mode='test', text_field=text_field)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
                          collate_fn=PadCollate())
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
                        collate_fn=PadCollate())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
                        collate_fn=PadCollate())


# Models (create model according to text embedding)
if embed_type == 'use':
    # For USE (Universal Sentence Embeddings)
    model_name = 'img_use_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
else:
    # For Glove and Fasttext Embeddings
    model_name = 'img_lstm_glove_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(use=False, hidden_size=300, embedding_length=word_embeddings.shape[1]).to(device)


optimizer = optim.Adam([
    {'params': combined_model.img_model.parameters(), 'lr': img_lr},
    {'params': combined_model.text_model.parameters(), 'lr': text_lr}],
    lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

print("Total Params", sum(p.numel() for p in combined_model.parameters() if p.requires_grad))
print("Img Model", sum(p.numel() for p in combined_model.img_model.parameters() if p.requires_grad))
print("Text Model", sum(p.numel() for p in combined_model.text_model.parameters() if p.requires_grad))

# Logger
logger = Logger(model_name=model_name, data_name='cosmos', log_path=os.path.join(BASE_DIR, 'tf_logs', model_name))


def train_model(epoch):
    """
        Performs one training epoch and updates the weight of the current model

        Args:
            epoch(int): Current epoch number

        Returns:
            None
    """
    train_loss = 0.
    total = 0.
    correct = 0.
    combined_model.train()
    # Training loop
    for batch_idx, (img, text_match, text_diff, seq_len_match, seq_len_diff, bboxes, bbox_classes) in enumerate(
            tqdm(train_loader)):
        text_match, text_diff = process_text_embedding(text_match, text_diff)
        batch = len(img)
        with torch.set_grad_enabled(True):
            z_img, z_t_match, z_t_diff = combined_model(img, text_match, text_diff, batch, seq_len_match, seq_len_diff,
                                                        bboxes, bbox_classes)
            loss = margin_loss_text_combined(z_img, z_t_match, z_t_diff)
            loss.backward()
            train_loss += float(loss.item())
            optimizer.step()
            optimizer.zero_grad()  # clear gradients for this training step
            correct += get_match_vs_no_match_acc(z_img, z_t_match, z_t_diff)
            total += batch
            torch.cuda.empty_cache()
            del img, text_match, text_diff, seq_len_match, seq_len_diff, bboxes, bbox_classes

    # Calculate loss and accuracy for current epoch
    logger.log(mode="train", scalar_value=train_loss / len(train_loader), epoch=epoch, scalar_name='loss')
    logger.log(mode="train", scalar_value=correct / total, epoch=epoch, scalar_name='accuracy')

    print(' Train Epoch: {} Loss: {:.4f} Acc: {:.2f} '.format(epoch, train_loss / len(train_loader), correct / total))


def evaluate_model(epoch):
    """
        Performs one validation epoch and computes loss and accuracy on the validation set

        Args:
            epoch (int): Current epoch number

        Returns:
            val_loss (float): Average loss on the validation set
    """
    combined_model.eval()
    val_loss = 0.
    total = 0.
    correct = 0.
    with torch.no_grad():
        for batch_idx, (img, text_match, text_diff, seq_len_match, seq_len_diff, bboxes, bbox_classes) in enumerate(
                tqdm(val_loader, desc='')):
            text_match, text_diff = process_text_embedding(text_match, text_diff)
            batch = len(img)
            z_img, z_t_match, z_t_diff = combined_model(img, text_match, text_diff, batch, seq_len_match, seq_len_diff,
                                                        bboxes, bbox_classes)
            loss = margin_loss_text_combined(z_img, z_t_match, z_t_diff)
            val_loss += float(loss.item())
            correct += get_match_vs_no_match_acc(z_img, z_t_match, z_t_diff)
            total += batch
            torch.cuda.empty_cache()
            del img, text_match, text_diff, seq_len_match, seq_len_diff, bboxes, bbox_classes

        logger.log(mode="val", scalar_value=val_loss / len(val_loader), epoch=epoch, scalar_name='loss')
        logger.log(mode="val", scalar_value=correct / total, epoch=epoch, scalar_name='accuracy')

        print(' Val Epoch: {} Avg loss: {:.4f} Acc: {:.2f}'.format(epoch, val_loss / len(val_loader), correct / total))
    return val_loss


def train_joint_model():
    """
        Performs training and validation on the dataset
    """
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(BASE_DIR + 'models/' + model_name + '.pt')
        combined_model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        combined_model.eval()
        best_loss = eval_validation_loss()
    except:
        best_loss = np.Inf
    early_stop = False
    counter = 0
    for epoch in range(1, epochs + 1):
        # Training epoch
        train_model(epoch)
        # Validation epoch
        avg_test_loss = evaluate_model(epoch)
        scheduler.step(avg_test_loss)
        if avg_test_loss <= best_loss:
            counter = 0
            best_loss = avg_test_loss
            torch.save(combined_model.state_dict(), 'models/' + model_name + '.pt')
            print("Best model saved/updated..")
            torch.cuda.empty_cache()
        else:
            counter += 1
            if counter >= patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            break


# Test with Match vs Non Match Accuracy
def test_match_accuracy():
    """
        Once the model is trained, it is used to evaluate the how accurately the captions align with the objects in the image
    """
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(BASE_DIR + 'models_final/' + model_name + '.pt')
        combined_model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        combined_model.eval()
        correct = 0.
        total = 0.
        with torch.no_grad():
            for i, (img, text_match, text_diff, seq_len_match, seq_len_diff, bboxes, bbox_classes) in enumerate(
                    tqdm(val_loader, desc='')):
                text_match, text_diff = process_text_embedding(text_match, text_diff)
                batch = len(img)
                z_img, z_t_match, z_t_diff = combined_model(img, text_match, text_diff, batch, seq_len_match,
                                                            seq_len_diff, bboxes, bbox_classes)
                correct += get_match_vs_no_match_acc(z_img, z_t_match, z_t_diff)
                total += batch
                torch.cuda.empty_cache()
                del img, text_match, text_diff, seq_len_match, seq_len_diff, bboxes, bbox_classes
        print('Accuracy : ', correct / total)
    except Exception as e:
        print(e)
        exit()


def eval_validation_loss():
    """
        Computes validation loss on the saved model, useful to resume training for an already saved model
    """
    val_loss = 0.
    with torch.no_grad():
        for batch_idx, (img, text_match, text_diff, seq_len_match, seq_len_diff, bboxes, bbox_classes) in enumerate(
                tqdm(val_loader, desc='')):
            text_match, text_diff = process_text_embedding(text_match, text_diff)
            batch = len(img)
            z_img, z_t_match, z_t_diff = combined_model(img, text_match, text_diff, batch, seq_len_match, seq_len_diff,
                                                        bboxes, bbox_classes)
            loss = margin_loss_text_combined(z_img, z_t_match, z_t_diff)
            val_loss += loss.item()
            torch.cuda.empty_cache()
            del img, text_match, text_diff, seq_len_match, seq_len_diff, bboxes, bbox_classes
        print(' Val Avg loss: {:.4f}'.format(val_loss / len(val_loader)))
    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mode', type=str, default='test',
                        help="mode, {'" + "train" + "', '" + "eval" + "'}")
    args = parser.parse_args()
    if args.mode == 'train':
        train_joint_model()
    elif args.mode == 'eval':
        test_match_accuracy()
