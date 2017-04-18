import os

from keras import backend as K
from keras.engine import Input, Model
from keras.engine.topology import Container, Layer
from keras.layers import Lambda, Concatenate, Flatten, Dense, Reshape, Convolution2D, UpSampling2D

from began.config import BEGANConfig


def build_model(config: BEGANConfig):
    K.set_image_data_format('channels_last')

    autoencoder, autoencoder_not_trainable = build_autoencoder(config)
    generator = build_generator(config)
    discriminator = build_discriminator(config, autoencoder)

    return autoencoder, autoencoder_not_trainable, generator, discriminator


def load_model_weight(model: Container, weight_file):
    if os.path.exists(weight_file):
        print("loading model weight: %s" % weight_file)
        model.load_weights(weight_file)


def build_autoencoder(config: BEGANConfig):
    n_filters = config.n_filters
    hidden_size = config.hidden_size

    dx = image_input = Input((config.image_height, config.image_width, 3))

    dx = convolution_image_for_encoding(dx, n_filters, strides=(2, 2))      # output: (N, 32, 32, n_filters)
    dx = convolution_image_for_encoding(dx, n_filters * 2, strides=(2, 2))  # output: (N, 16, 16, n_filters*2)
    dx = convolution_image_for_encoding(dx, n_filters * 3, strides=(2, 2))  # output: (N, 8, 8, n_filters*3)
    dx = convolution_image_for_encoding(dx, n_filters * 4, strides=(1, 1))  # output: (N, 8, 8, n_filters*4)
    dx = Flatten()(dx)
    hidden = Dense(hidden_size, activation='linear')(dx)
    image_output = build_decoder_layer(config, hidden)

    autoencoder = Container(image_input, image_output, name="autoencoder")
    autoencoder_not_trainable = Container(image_input, image_output, name="autoencoder_not_trainable")
    autoencoder_not_trainable.trainable = False

    return autoencoder, autoencoder_not_trainable


def build_generator(config: BEGANConfig):
    hidden_size = config.hidden_size
    z_input = Input((hidden_size, ))
    image_output = build_decoder_layer(config, z_input)
    generator = Model(z_input, image_output)
    return generator


def build_discriminator(config: BEGANConfig, autoencoder: Container):
    """
    
    Keras Model class is able to have several inputs/outputs. 
    But, loss functions should be defined each other, and the loss function cannot reference other inputs/outputs.
    For computing loss, two inputs/outputs are concatenated.
    """
    # IN Shape: [ImageHeight, ImageWidth, (real data(3 channels) + generated data(3 channels))]
    in_out_shape = (config.image_height, config.image_width, 3 * 2)
    all_input = Input(in_out_shape)

    # Split Input Data
    data_input = Lambda(lambda x: x[:, :, :, 0:3], output_shape=(config.image_height, config.image_width, 3))(all_input)
    generator_input = Lambda(lambda x: x[:, :, :, 3:6], output_shape=(config.image_height, config.image_width, 3))(all_input)

    # use same autoencoder(weights are shared)
    data_output = autoencoder(data_input)  # (bs, row, col, ch)
    generator_output = autoencoder(generator_input)

    # concatenate output to be same shape of input
    all_output = Concatenate(axis=-1)([data_output, generator_output])

    discriminator = DiscriminatorModel(all_input, all_output, name="discriminator")
    return discriminator


def build_decoder_layer(config: BEGANConfig, input_layer):
    """
    generator and decoder( of discriminator) have same network structure, but don't share weights.
    This function takes different input layer, flow another network, and return different output layer.
    """
    n_filters = config.n_filters

    dx = input_layer  # (64, )
    dx = Dense((8*8*n_filters), activation='linear')(dx)
    dx = Reshape((8, 8, n_filters))(dx)
    dx = convolution_image_for_decoding(dx, n_filters, upsample=True)   # output: (N, 16, 16, n_filters)
    dx = convolution_image_for_decoding(dx, n_filters, upsample=True)   # output: (N, 32, 32, n_filters)
    dx = convolution_image_for_decoding(dx, n_filters, upsample=True)   # output: (N, 64, 64, n_filters)
    dx = convolution_image_for_decoding(dx, n_filters, upsample=False)  # output: (N, 64, 64, n_filters)
    image_output = Convolution2D(3, (3, 3), padding="same", activation="linear")(dx)  # output: (N, 64, 64, 3), activation shuold be linear?
    return image_output


def convolution_image_for_encoding(in_x, filters, strides=(1, 1)):
    x = Convolution2D(filters, (3, 3), activation="elu", padding="same")(in_x)
    x = Convolution2D(filters, (3, 3), activation="elu", padding="same")(x)
    x = Convolution2D(filters, (3, 3), activation="elu", padding="same", strides=strides)(x)
    return x


def convolution_image_for_decoding(in_x, filters, upsample=None):
    x = Convolution2D(filters, (3, 3), activation="elu", padding="same")(in_x)
    x = Convolution2D(filters, (3, 3), activation="elu", padding="same")(x)
    x = Convolution2D(filters, (3, 3), activation="elu", padding="same")(x)
    if upsample:
        x = UpSampling2D()(x)
    return x


class DiscriminatorModel(Model):
    """Model which collects updates from loss_func.updates"""

    @property
    def updates(self):
        updates = super().updates
        if hasattr(self, 'loss_functions'):
            for loss_func in self.loss_functions:
                if hasattr(loss_func, 'updates'):
                    updates += loss_func.updates
        return updates
