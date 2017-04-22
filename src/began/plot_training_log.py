# coding: utf8
from began.config import BEGANConfig
import pandas as pd
import matplotlib.pyplot as plt


def main():
    config = BEGANConfig()
    plot_training_log(config)


def plot_training_log(config: BEGANConfig):
    df = pd.read_csv(config.training_log)
    df = df.rolling(window=1000, center=False).mean()
    df.plot(y=["m_global", "k", "loss_discriminator", "loss_generator", "loss_real_x", "loss_gen_x"],
            title="Moving Average(1000): Training History")
    plt.savefig(config.training_graph)


if __name__ == '__main__':
    main()
