# coding: utf8
import os

from began.config import BEGANConfig
from began.models import build_generator, load_model_weight
import numpy as np
from PIL import Image


def main():
    config = BEGANConfig()
    generate(config, "main")


def generate(config: BEGANConfig, dir_name, generator=None):
    output_dir = "%s/%s" % (config.generated_dir, dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if generator is None:
        generator = build_generator(config)
        load_model_weight(generator, config.generator_weight_filename)

    np.random.seed(999)
    num_image = 20
    z = np.random.uniform(-1, 1, config.hidden_size * num_image).reshape((num_image, config.hidden_size))
    x = generator.predict(z)
    images = np.clip(x * 255, 0, 255).astype("uint8")
    print((x.shape, np.min(x), np.max(x)))

    for i in range(num_image):
        image_x = images[i]
        img = Image.fromarray(image_x)
        img.save("%s/gen_%03d.jpg" % (output_dir, i))


if __name__ == '__main__':
    main()
