import os, sys
import tensorflow as tf
from waveletdetection.dataset.signal_dataset import SignalDataset
from waveletdetection.model.cnn_model import CNNModel
from waveletdetection.trainer.trainer import Trainer


def main_train(config):

    dataset = SignalDataset(config)

    session = tf.Session()

    # TODO: init the right model from config
    model = CNNModel(config)

    trainer = Trainer(session, model, dataset, config)

    trainer.train()


if __name__ == '__main__':

    from waveletdetection.config.config_reader import ConfigReader

    config_path = '/Users/adamzvada/Documents/Avast/WaveletDetection/config/test.yml'

    main_train(ConfigReader(config_path))
