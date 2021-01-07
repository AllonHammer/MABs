import tensorflow as tf
import multiprocessing
from joblib import Parallel, delayed
from mabs import *
from runtime import naive_training, mab_training, run_parallel
from utils import plot_all_in_one, argument_parser

if __name__ == "__main__":
    print('Is GPU available', tf.test.is_gpu_available())
    print(tf.config.list_physical_devices('GPU'))

    args = argument_parser()
    batch_size = args.batch
    epochs = args.epochs
    gamma = args.gamma
    eta = args.eta
    epsilon = args.epsilon
    mab_type = args.type
    parallel = args.parallel
    if parallel:
        print('Running in parallel...')
        mabs_types = ['EpsilonGreedy', 'Random', 'UCB1', 'Thompson', 'EXP3', 'EXP3IX', 'FTL']
        parameters = {'batch_size': batch_size, 'epochs': epochs, 'gamma': gamma, 'eta': eta, 'epsilon': epsilon}
        num_cores = multiprocessing.cpu_count()
        processed_list = Parallel(n_jobs=num_cores)(delayed(run_parallel)(mab, parameters) for mab in mabs_types)
        # For MAB algorithms: each "epoch" ingests "batch_size" samples
        # For naive training: each epoch ingests the entire data set (60000 samples)
        # In order to get aligned samples the naive training should get much less "epochs"
        # Example: batch_size = 150, epochs = 100 , total samples ingested by MAB: 150000
        # Example: naive training should get 2-3 epochs (60,000 * 2.5) = 150000
        naive_epochs = int(np.ceil((epochs * batch_size) / 60000))
        naive_training(naive_epochs, batch_size)
        plot_all_in_one()
        exit()

    if mab_type == 'EpsilonGreedy':
        mab = MabEpsilonGreedy(epsilon)
    elif mab_type == 'Random':
        mab = MabRandomArm()
    elif mab_type == 'UCB1':
        mab = MabUcb1()
    elif mab_type == 'Thompson':
        mab = MabThompsonSampling()
    elif mab_type == 'EXP3':
        mab = MabExp3(gamma)
    elif mab_type == 'EXP3IX':
        mab = MabExp3Ix(gamma, eta)
    elif mab_type == 'FTL':
        mab = MabFtl(eta)
    elif mab_type == 'Benchmark':
        naive_training(epochs, batch_size)
        exit()
    elif mab_type == 'Plot':
        plot_all_in_one()
        exit()
    else:
        print('Wrong MAB type: {}. Must be in: {}'.format(mab_type, ['EpsilonGreedy', 'Random', 'UCB1',
                                                                     'Thompson', 'EXP3', 'EXP3IX', 'FTL', 'Benchmark'
                                                                     'Plot']))
        exit()
    mab_training(mab, epochs, batch_size)


