## Getting Started

- **Source Code:**   `$ git clone` this repo and install the Python dependencies from `requirements.txt`
- **Dataset** Download the dataset by filling out the
   form [here](https://docs.google.com/forms/d/10Nke6m8MvCxP7hoJQ_k-mtiejbXtE0RliX9w_8pooLQ/edit).

This document provides a brief intro of the usage of this repo.

### Training & Evaluation

This code is based on [Detectron2](https://github.com/facebookresearch/detectron2) to extract features from objects present in the image. Please setup and install Detectron2 first if you wish to use our feature detector for images. The minimal changes to be done to Detectron2 source code to extract object features are added to [detectron2_changes](https://github.com/shivangi-aneja/COSMOS/tree/main/detectron2_changes) directory. Navigate to Detectron2 source code directory and simply copy and replace these files. Consider setting up Detectron2 inside the home directory, it works seamlessly without doing many changes.                                 
All the training parameters are configured via [utils/config.py](https://github.com/shivangi-aneja/COSMOS/blob/main/utils/config.py). Specify paths, hyperparameters, text-embeddings, threshold values, etc in the `config.py` file. Model names are specifed in the trainer script itself. Configure these parameters according to your need and start training.     
To train the model, execute the following command:
```
    python trainer_scipt.py -m train
```      


Once training is finished, then to evaluate the model with Match vs No-Match Accuracy, execute the following command:
``` 
    python trainer_scipt.py -m eval
```


### Inference with Pre-trained Models

1. Pick a model configuration from `utils.config.py` file.
2. We provide `evaluate_ooc.py` that can be used to evaluate the trained models for out-of-context detection task. 

Run the file :
```
python evaluate_ooc.py
```
The configs are made for training as well as evaluation, therefore we need to specify these arguments.
