import tensorflow as tf
from time import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tqdm import tqdm
from utils import *
from mabs import *

def create_cnn_model():
    """
    Simple CNN model:
          Input --> CONV (28 filters 3X3) --> MaxPool (2X2) --> Flatten --> Dense + Relu (128) ---> Dense + Softmax(10)
    :return: keras.models.Sequential()
    """
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def naive_training(epochs, batch_size, plot=False):
    """
    Train the network regularly. Without MABs

    :param epochs: int one epoch iterates the entire data
    :param batch_size: int
    :param plot: boolean
    :return:
    """
    # Get data
    x_train, x_test, y_train, y_test = prepare_data()
    n_samples = x_train.shape[0]
    # Create and compile model
    model = create_cnn_model()
    # Initial evaluation
    init_acc = model.evaluate(x_test, y_test, verbose=0)[1]
    # Start Timer
    start_time = time()
    # Fit
    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    end_time = time() - start_time
    # Time elapsed per epoch
    times = [0] + [epoch * (end_time/epochs) for epoch in range(1, epochs+1)]
    times = [round(t) for t in times]
    # Losses per epoch
    losses = [init_acc] + hist.history['val_accuracy']
    # Cumulative samples ingested by network
    samples = [0] + [n_samples * epoch for epoch in range(1, epochs+1)]
    # Plot on graph
    eval = model.evaluate(x_test, y_test, verbose=0)
    print('Final evaluation is ', eval)
    if plot:
        plot_performance(losses, times, samples, file_name='Regular_Training', skip_ticks=1)
    # write results to csv
    results = {'samples': samples, 'times': times, 'losses': losses}
    save_results_to_csv(model_name='No MAB', data_dict=results)


def mab_training(mab, epochs, batch_size, plot=False):
    """
    Train the network with smart data sampling using MABs

    :param mab: an instance of MAB()
    :param epochs: int one epoch iterates only "batch_size" samples!
    :param batch_size: int
    :param plot: boolean
    :return:
    """

    # Create and compile model
    model = create_cnn_model()
    # Initial evaluation
    previous_acc = model.evaluate(mab.x_test, mab.y_test, verbose=0)[1]
    # Losses per epoch
    losses = [previous_acc]
    # Cumulative samples ingested by network
    samples = [0]
    # Time elapsed per epoch
    times = [0]
    start_time = time()
    for epoch in tqdm(range(1, epochs+1)):
        # Select arm (data cluster) according to MAB
        mab.select_arm()
        # Get a batch_size of data from this cluster
        x,y = mab.get_batch(batch_size)
        # Fit once on this batch
        model.fit(x=x, y=y, epochs=1, batch_size=batch_size, verbose=0)
        # Observe loss on test set
        new_acc = model.evaluate(mab.x_test, mab.y_test, verbose=0)[1]
        # Calculate reward (improvement from previous loss)
        reward = max(new_acc - previous_acc, 0)  # keep rewards between 0 and 1
        # Update MABs weights
        mab.update(reward)
        # Append lists for evaluation
        losses.append(new_acc)
        samples.append(epoch*batch_size)
        times.append(round(time()- start_time))
        # Update current loss
        previous_acc = new_acc
    # plot on graph
    eval = model.evaluate(mab.x_test, mab.y_test, verbose=0)
    print('Final evaluation is ', eval)
    if plot:
        plot_performance(losses, times, samples, file_name=type(mab).__name__)
    # write results to csv
    results = {'samples': samples, 'times': times, 'losses' : losses}
    save_results_to_csv(model_name=type(mab).__name__, data_dict=results)


def run_parallel(mab_type, params):
    """
    Wrapper function for parallel execution

    :param mab_type: str
    :param params: dict <param: value>
    :return:
    """
    epochs = params['epochs']
    batch_size = params['batch_size']
    epsilon = params['epsilon'] if 'epsilon' in params else 0.1
    eta = params['eta'] if 'eta' in params else 0.1
    gamma = params['gamma'] if 'gamma' in params else 0.1

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

    else:
        print('Wrong MAB type: {}. Must be in: {}'.format(mab_type, ['EpsilonGreedy', 'Random', 'UCB1',
                                                                     'Thompson', 'EXP3', 'EXP3IX', 'FTL',
                                                                     'Benchmark']))
        exit()

    mab_training(mab, epochs, batch_size, plot=False)