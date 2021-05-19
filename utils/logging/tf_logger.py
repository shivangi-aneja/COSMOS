"""
    Tensorflow logger for monitoring train/val curves
"""

import errno
import os
from torch.utils.tensorboard import SummaryWriter
import torch


class Logger:

    def __init__(self, model_name, data_name, log_path):
        """
            Initializes the logger object for computing loss/accuracy curves
            Args:
                model_name (str): The name of the model for which training needs to be monitored
                data_name (str): Dataset name
                log_path (str): Base path for logging
        """
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)

        # TensorBoard
        self.train_writer = SummaryWriter(log_dir=log_path+'/train/', comment=self.comment)
        self.val_writer = SummaryWriter(log_dir=log_path+'/val/', comment=self.comment)

    def log(self, mode, scalar_value, epoch, scalar_name='error'):
        """
            Logs the scalar value passed for train and val epoch
            Args:
                mode (str): train/val
                scalar_value (float): loss/accuracy value to be logged
                epoch (int): epoch number
                scalar_name (str): name of scalar to be logged
            Returns:
                None
        """

        if isinstance(scalar_value, torch.autograd.Variable):
            scalar_value = scalar_value.data.cpu().numpy()

        if mode == 'train':
            self.train_writer.add_scalar(self.comment + '_' + scalar_name, scalar_value, epoch)
        if mode == 'val':
            self.val_writer.add_scalar(self.comment + '_' + scalar_name, scalar_value, epoch)
