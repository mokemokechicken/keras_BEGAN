# coding: utf8
import os

THIS_DIR = os.path.dirname(__file__)


class BEGANConfig:
    def __init__(self):
        self.project_dir = '%s/../..' % THIS_DIR
        self.resource_dir = '%s/resource' % self.project_dir
        self.dataset_dir = self.resource_dir
        self.generated_dir = "%s/generated" % self.project_dir
        self.dataset_filename = "%s/dataset.pkl" % (self.dataset_dir, )
        self.image_width = 64
        self.image_height = 64
        self.n_filters = 128
        self.hidden_size = 64
        self.initial_k = 0
        self.gamma = 0.5
        self.lambda_k = 0.001
        self.batch_size = 16
        self.initial_lr = 0.0001
        self.min_lr = 0.00001
        self.lr_decay_rate = 0.9

        self.autoencoder_weight_filename = '%s/autoencoder.hd5' % (self.dataset_dir, )
        self.generator_weight_filename = '%s/generator.hd5' % (self.dataset_dir, )
        self.discriminator_weight_filename = '%s/discriminator.hd5' % (self.dataset_dir, )
        self.training_log = '%s/training_log.csv' % (self.generated_dir, )
        self.training_graph = "%s/training.png" % (self.generated_dir, )
