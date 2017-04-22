import os

from keras import backend as K
from keras.engine import Input, Model
from keras.engine.topology import Container, Layer
from keras.layers import Lambda, Concatenate, Flatten, Dense, Reshape, Convolution2D, UpSampling2D

from began.config import BEGANConfig


def build_model(config: BEGANConfig):
    K.set_image_data_format('channels_last')

    autoencoder = build_autoencoder(config)
    generator = build_generator(config)
    discriminator = build_discriminator(config, autoencoder)

    return autoencoder, generator, discriminator


def load_model_weight(model: Container, weight_file):
    if os.path.exists(weight_file):
        print("loading model weight: %s" % weight_file)
        model.load_weights(weight_file)


def build_autoencoder(config: BEGANConfig, name="autoencoder"):
    encoder = build_encoder(config, name="%s/encoder" % name)
    decoder = build_decoder(config, name="%s/decoder" % name)
    autoencoder = Container(encoder.inputs, decoder(encoder.outputs), name=name)
    return autoencoder


def build_generator(config: BEGANConfig):
    decoder = build_decoder(config, name="generator_decoder")
    generator = Model(decoder.inputs, decoder.outputs, name="generator")
    return generator


def build_encoder(config: BEGANConfig, name="encoder"):
    n_filters = config.n_filters
    hidden_size = config.hidden_size
    n_layer = config.n_layer_in_conv

    dx = image_input = Input((config.image_height, config.image_width, 3))

    # output: (N, 32, 32, n_filters)
    dx = convolution_image_for_encoding(dx, n_filters, strides=(2, 2), name="%s/L1" % name, n_layer=n_layer)

    # output: (N, 16, 16, n_filters*2)
    dx = convolution_image_for_encoding(dx, n_filters * 2, strides=(2, 2), name="%s/L2" % name, n_layer=n_layer)

    # output: (N, 8, 8, n_filters*3)
    dx = convolution_image_for_encoding(dx, n_filters * 3, strides=(2, 2), name="%s/L3" % name, n_layer=n_layer)

    # output: (N, 8, 8, n_filters*4)
    dx = convolution_image_for_encoding(dx, n_filters * 4, strides=(1, 1), name="%s/L4" % name, n_layer=n_layer)

    dx = Flatten()(dx)
    hidden = Dense(hidden_size, activation='linear', name="%s/Dense" % name)(dx)

    encoder = Container(image_input, hidden, name=name)
    return encoder


def build_decoder(config: BEGANConfig, name):
    """
    generator and decoder( of discriminator) have same network structure, but don't share weights.
    This function takes different input layer, flow another network, and return different output layer.
    """
    n_filters = config.n_filters
    n_layer = config.n_layer_in_conv

    dx = input_z = Input((64, ))
    dx = Dense((8*8*n_filters), activation='linear', name="%s/Dense" % name)(dx)
    dx = Reshape((8, 8, n_filters))(dx)

    # output: (N, 16, 16, n_filters)
    dx = convolution_image_for_decoding(dx, n_filters, upsample=True, name="%s/L1" % name, n_layer=n_layer)

    # output: (N, 32, 32, n_filters)
    dx = convolution_image_for_decoding(dx, n_filters, upsample=True, name="%s/L2" % name, n_layer=n_layer)

    # output: (N, 64, 64, n_filters)
    dx = convolution_image_for_decoding(dx, n_filters, upsample=True, name="%s/L3" % name, n_layer=n_layer)

    # output: (N, 64, 64, n_filters)
    dx = convolution_image_for_decoding(dx, n_filters, upsample=False, name="%s/L4" % name, n_layer=n_layer)

    # output: (N, 64, 64, 3)
    image_output = Convolution2D(3, (3, 3), padding="same", activation="linear", name="%s/FinalConv" % name)(dx)
    decoder = Container(input_z, image_output, name=name)
    return decoder


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


def convolution_image_for_encoding(x, filters, strides=(1, 1), name=None, n_layer=2):
    for i in range(1, n_layer):
        x = Convolution2D(filters, (3, 3), activation="elu", padding="same", name="%s/Conv%d" % (name, i))(x)

    x = Convolution2D(filters, (3, 3), activation="elu", padding="same", strides=strides,
                      name="%s/Conv%d" % (name, n_layer))(x)
    return x


def convolution_image_for_decoding(x, filters, upsample=None, name=None, n_layer=2):
    for i in range(1, n_layer+1):
        x = Convolution2D(filters, (3, 3), activation="elu", padding="same", name="%s/Conv%d" % (name, i))(x)
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
