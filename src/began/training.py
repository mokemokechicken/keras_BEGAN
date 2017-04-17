# coding: utf8
import pickle
from datetime import datetime

import keras.backend as K
import numpy as np
from keras.engine.topology import Container
from keras.optimizers import Adam

from began.config import BEGANConfig
from began.generate_image import generate
from began.models import build_model, load_model_weight

ESC = chr(0x1b)
UP = ESC+"[1A"


def main():
    config = BEGANConfig()
    training(config, epochs=500)


def training(config: BEGANConfig, epochs=3):
    # loading dataset, and let values to 0.0 ~ 1.0
    with open(config.dataset_filename, 'rb') as f:
        dataset = pickle.load(f)  # type: np.ndarray
        dataset = dataset / 255.0
        info(dataset.shape)

    batch_size = config.batch_size

    # building model and loading weight(if exists)
    autoencoder, autoencoder_not_trainable, generator, discriminator = build_model(config)
    load_model_weight(autoencoder, config.autoencoder_weight_filename)
    load_model_weight(generator, config.generator_weight_filename)
    load_model_weight(discriminator, config.discriminator_weight_filename)

    loss_d = DiscriminatorLoss(config.initial_k)  # special? loss object for BEGAN
    discriminator.compile(optimizer=Adam(), loss=loss_d)
    generator.compile(optimizer=Adam(), loss=create_generator_loss(autoencoder))
    lr_decay_step = 0
    last_m_global = np.Inf
    log_recorder = LogRecorder(config.training_log)

    np.random.seed(999)
    for ep in range(1, epochs+1):
        # generate Z layer values for discriminator and generator
        zd = np.random.uniform(-1, 1, (len(dataset), config.hidden_size))
        zg = np.random.uniform(-1, 1, (len(dataset), config.hidden_size))

        # shuffle dataset index
        index_order = np.arange(len(dataset))
        np.random.shuffle(index_order)

        # set Learning Rate
        lr = max(config.initial_lr * (config.lr_decay_rate ** lr_decay_step), config.min_lr)
        K.set_value(generator.optimizer.lr, lr)
        K.set_value(discriminator.optimizer.lr, lr)
        m_global_history = []
        info("LearningRate=%.7f" % lr)
        batch_len = len(dataset)//batch_size

        for b_idx in range(batch_len):
            index_list = index_order[b_idx*batch_size:(b_idx+1)*batch_size]

            # training discriminator
            in_x1 = dataset[index_list]  # (bs, row, col, ch)
            in_x2 = generator.predict_on_batch(zd[index_list])
            in_x = np.concatenate([in_x1, in_x2], axis=-1)  # (bs, row, col, ch*2)
            loss_discriminator = discriminator.train_on_batch(in_x, in_x)

            # training generator
            in_x1 = zg[index_list]
            loss_generator = generator.train_on_batch(in_x1, np.zeros_like(in_x2))  # y_true is meaningless

            # record M-Global
            m_global_history.append(loss_d.m_global)
            if b_idx > 0:
                print(UP + UP)
            log_info = dict(
                epoch=ep,
                batch_index=b_idx,
                batch_len=batch_len,
                m_global=loss_d.m_global,
                loss_discriminator=loss_discriminator,
                loss_generator=loss_generator,
                loss_real_x=loss_d.loss_real_x,
                loss_gen_x=loss_d.loss_gen_x,
                k=loss_d.k,
                lr=lr,
            )
            info("ep=%(epoch)s, b_idx=%(batch_index)s/%(batch_len)s, MGlobal=%(m_global).5f, "
                 "Loss(D)=%(loss_discriminator).5f, Loss(G)=%(loss_generator).5f, Loss(X)=%(loss_real_x).5f, "
                 "Loss(G(Zd))=%(loss_gen_x).5f, K=%(k).6f" % log_info)
            log_recorder.write(**log_info)

        m_global = np.average(m_global_history)
        if last_m_global <= m_global:  # decay LearningRate
            lr_decay_step += 1
        last_m_global = m_global

        # Save Model Weight in each epoch
        autoencoder.save_weights(config.autoencoder_weight_filename)
        generator.save_weights(config.generator_weight_filename)
        discriminator.save_weights(config.discriminator_weight_filename)

        # Generate Image in each epoch for fun
        generate(config, "ep%03d" % ep, generator)


def create_generator_loss(autoencoder: Container):
    def generator_loss(y_true, y_pred):
        y_pred_dash = autoencoder(y_pred)
        return K.mean(K.abs(y_pred - y_pred_dash), axis=[1, 2, 3])
    return generator_loss


class DiscriminatorLoss:
    __name__ = 'discriminator_loss'

    def __init__(self, initial_k=0, lambda_k=0.001, gamma=0.5):
        self.lambda_k = lambda_k
        self.gamma = gamma
        self.k_var = K.variable(initial_k, dtype=K.floatx(), name="discriminator_k")
        self.m_global_var = K.variable(0, dtype=K.floatx(), name="m_global")
        self.loss_real_x_var = K.variable(0, name="loss_real_x")  # for observation
        self.loss_gen_x_var = K.variable(0, name="loss_gen_x")    # for observation
        self.updates = []

    def __call__(self, y_true, y_pred):  # y_true, y_pred shape: (BS, row, col, ch * 2)
        data_true, generator_true = y_true[:, :, :, 0:3], y_true[:, :, :, 3:6]
        data_pred, generator_pred = y_pred[:, :, :, 0:3], y_pred[:, :, :, 3:6]
        loss_data = K.mean(K.abs(data_true - data_pred), axis=[1, 2, 3])
        loss_generator = K.mean(K.abs(generator_true - generator_pred), axis=[1, 2, 3])
        ret = loss_data - self.k_var * loss_generator

        # for updating values in each epoch, use `updates` mechanism
        # DiscriminatorModel collects Loss Function's updates attributes
        mean_loss_data = K.mean(loss_data)
        mean_loss_gen = K.mean(loss_generator)

        # update K
        new_k = self.k_var + self.lambda_k * (self.gamma * mean_loss_data - mean_loss_gen)
        new_k = K.clip(new_k, 0, 1)
        self.updates.append(K.update(self.k_var, new_k))

        # calculate M-Global
        m_global = mean_loss_data + K.abs(self.gamma * mean_loss_data - mean_loss_gen)
        self.updates.append(K.update(self.m_global_var, m_global))

        # let loss_real_x mean_loss_data
        self.updates.append(K.update(self.loss_real_x_var, mean_loss_data))

        # let loss_gen_x mean_loss_gen
        self.updates.append(K.update(self.loss_gen_x_var, mean_loss_gen))

        return ret

    @property
    def k(self):
        return K.get_value(self.k_var)

    @property
    def m_global(self):
        return K.get_value(self.m_global_var)

    @property
    def loss_real_x(self):
        return K.get_value(self.loss_real_x_var)

    @property
    def loss_gen_x(self):
        return K.get_value(self.loss_gen_x_var)


def info(msg):
    now = datetime.now()
    print("%s: %s" % (now, msg))


class LogRecorder:
    def __init__(self, log_filename):
        self.file_out = open(log_filename, "wt")
        self.columns = None

    def write(self, **kwargs):
        if not self.columns:
            self.columns = list(sorted(kwargs.keys()))
            self.file_out.write(",".join(self.columns) + "\n")
        values = [str(kwargs.get(x, "")) for x in self.columns]
        self.file_out.write(",".join(values) + "\n")
        self.file_out.flush()


if __name__ == '__main__':
    main()
