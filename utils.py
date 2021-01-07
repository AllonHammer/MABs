import argparse
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, help='batch_size', default=100)
    parser.add_argument('-n', '--epochs', type=int, help='epochs', default=5)
    parser.add_argument('-e', '--eta', type=float, help='eta', default=0.1)
    parser.add_argument('-g', '--gamma', type=float, help='gamma', default=0.1)
    parser.add_argument('-s', '--epsilon', type=float, help='epsilon', default=0.01)
    parser.add_argument('-t', '--type', type=str, help='mab type', default='Benchmark')
    parser.add_argument('-p', '--parallel', default=False, help='Run in parallel', action='store_true')

    return parser.parse_args()


def plot_performance(losses, times, samples, file_name, skip_ticks=5):
    """
    Plots the evaluation of a single run

    :param losses: list of losses as a function of time
    :param times: list of time elapsed since first epoch (seconds)
    :param samples: list of number of samples ingested by network as a function of time
    :param file_name: str path to save fig
    :param skip_ticks: int how many ticks to skip in secondary axis
    :return:
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(samples, losses)
    ax1.set_ylabel('Test set accuracy')
    ax1.set_xlabel("Samples ingested by network")

    ax2.set_xlim(0, samples[-1])
    ax2.set_xticks(samples[0::skip_ticks])
    ax2.set_xticklabels(times[0::skip_ticks])
    ax2.set_xlabel("Time passed in seconds")
    plt.savefig('./{}.png'.format(file_name))
    plt.show()


def plot_all_in_one(first_n_percent= 1.0):
    """
    Plots the evaluation of all tests and compared to benchmark
    :return:
    """
    skip_ticks = 5
    path = './results'  # use your path
    all_files = glob.glob(path + "/*.csv")

    df_lst = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df = df[['model', 'samples', 'times', 'losses']]
        df_lst.append(df)

    df = pd.concat(df_lst, axis=0, ignore_index=True)
    model_names = df['model'].unique()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_ylabel('Test set accuracy', fontweight='bold')
    ax1.set_xlabel("Samples ingested by network", fontweight='bold')

    for model in model_names:
        df_ = df[df['model'] == model]
        samples = df_.samples.values
        losses = df_.losses.values
        times = df_.times.values
        if model == 'No MAB':
            # extrapolate samples
            f = interp1d(samples, losses, kind='cubic')
            samples = np.arange(0, samples[-1], 100)
            max_loss = losses.max() - 0.01
            losses = f(samples)
            losses = np.clip(losses, 0, max_loss)
        # cutoff first_n_percent of rows
        cutoff = int(first_n_percent*len(samples))
        samples = samples[0:cutoff]
        losses = losses[0:cutoff]
        ax1.plot(samples, losses)


    ax1.legend(model_names, loc='lower right', prop={'size': 6})
    plt.title('Convergence Plot')
    plt.savefig('./results/{}.png'.format('Compare_all'))


def save_results_to_csv(model_name, data_dict):
    df = pd.DataFrame(data_dict, columns=['samples', 'times', 'losses'])
    df['model'] = model_name
    df.to_csv('./results/{}.csv'.format(model_name), index=False)



