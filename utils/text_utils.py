""" Utility function to retrieve and process textual data """

import spacy
import os
import pandas as pd
import torch
from torchtext import data
from utils.config import DATA_DIR, embed_type
from utils.common_utils import read_json_data
from nltk.stem.snowball import SnowballStemmer
# python3 -m spacy download en
spacy_en = spacy.load('en')
stemmer = SnowballStemmer(language="english")
from torchtext.vocab import GloVe, FastText


def get_text_metadata():
    """
        Returns word embeddings for glove/fasttext text embeddings, None for use model
    """
    if embed_type == 'use':
        return None, None, None
    text_field = data.Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
    captions = get_caption_list()
    preprocessed_caption = pd.DataFrame(captions, columns=['caption'])['caption'].apply(
        lambda x: text_field.preprocess(x))
    if embed_type == 'glove':
        text_field.build_vocab(preprocessed_caption, vectors=GloVe(name='6B', dim=300))
    elif embed_type == 'fasttext':
        text_field.build_vocab(preprocessed_caption, vectors=FastText(language='en'))
    word_embeddings = text_field.vocab.vectors
    vocab_size = len(text_field.vocab)
    print("Length of Text Vocabulary: " + str(vocab_size))
    print("Unique Word Vectors", torch.unique(text_field.vocab.vectors, dim=0).shape)
    print("Vector size of Text Vocabulary: ", word_embeddings.size())
    return text_field, word_embeddings, vocab_size


def get_caption_list():
    """
        Returns the captions from the dataset as a list

        Returns:
             captions (List[str]): List of captions
    """
    # train_data = read_json_data(os.path.join(DATA_DIR_OURS, 'annotations/val_data.json'))
    val_data = read_json_data(os.path.join(DATA_DIR, 'annotations', 'val_data.json'))[:30]
    # train_captions = retrieve_train_captions(train_data)
    val_captions = retrieve_captions(val_data)
    captions = val_captions #+ train_captions
    return captions


def retrieve_captions(data_list):
    """
        Reads the captions from a list and returns them

        Args:
            data_list (list<dict>): List of image metadata

        Returns:
            merged_captions (list<str>): List of captions from the dataset
    """
    caption_list = []
    for data in data_list:
        for article in data['articles']:
            caption_list.append(article['caption'])
    return caption_list


def tokenize(text):
    """
        Splits a sentence (default separator whitespace) and tokenizes it

        Args:
            text (str): Textual Caption

        Returns:
            list: List of tokenized words in the sentence

    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
