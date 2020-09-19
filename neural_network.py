import numpy as np
from utils import red_wine_quality, heart_failure_prediction, get_mse, scale_data
from plotter import Plotter
from sklearn.neural_network import MLPClassifier

def create_nn(x, y, max_iter, hidden_layer_sizes=(100,)):
    return MLPClassifier(
        solver='sgd',
        random_state=0,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        n_iter_no_change=1
    ).fit(x, y.values.flatten())

def create_nns(x, y, iter_counts, hidden_layer_sizes=(100,)):
    nns = []

    for count in iter_counts:
        print('NN {}'.format(count))
        nn = create_nn(x=x, y=y, max_iter=count, hidden_layer_sizes=hidden_layer_sizes)
        nns.append(nn)

    return nns

if __name__ == "__main__":
    # Split testing/training data and create trees
    hfp_x_train, hfp_y_train, hfp_x_test, hfp_y_test = heart_failure_prediction()
    rwq_x_train, rwq_y_train, rwq_x_test, rwq_y_test = red_wine_quality()

    # Scale data
    hfp_x_train, hfp_x_test = scale_data(hfp_x_train, hfp_x_test)
    rwq_x_train, rwq_x_test = scale_data(rwq_x_train, rwq_x_test)

    # Hidden layer sizes
    hfp_hidden_layer_sizes = (hfp_x_train.shape[1]**2, hfp_x_train.shape[1], hfp_x_train.shape[1]**2)
    rwq_hidden_layer_sizes = (rwq_x_train.shape[1]**2, rwq_x_train.shape[1], rwq_x_train.shape[1]**2)

    # Hyperparameters
    hfp_iter_counts = np.arange(1, 3100, 100, dtype=int)
    rwq_iter_counts = np.arange(1, 1100, 50, dtype=int)

    # Create Heart Failure Prediction NNs
    hfp_nns  = create_nns(hfp_x_train, hfp_y_train, hfp_iter_counts, hfp_hidden_layer_sizes)

    # Get training/testing accuracy for Heart Failure Prediction
    hfp_train_mse = get_mse(hfp_nns, hfp_x_train, hfp_y_train)
    hfp_test_mse = get_mse(hfp_nns, hfp_x_test, hfp_y_test)

    # Generate graph for Heart Failure Prediction
    plot = Plotter(
        name='Heart Failure Prediction', 
        learner='neural-network', 
        axes={ 'x': 'Number of weight updates', 'y': 'Error' }
    )
    plot.add_plot(hfp_iter_counts, hfp_train_mse, 'training data', 'None')
    plot.add_plot(hfp_iter_counts, hfp_test_mse, 'testing data', 'None')
    plot.save()

    # Create Red Wine Quality NNs
    rwq_nns = create_nns(rwq_x_train, rwq_y_train, rwq_iter_counts, rwq_hidden_layer_sizes)

    # Get training/testing accuracy for Red Wine Quality
    rwq_train_mse = get_mse(rwq_nns, rwq_x_train, rwq_y_train)
    rwq_test_mse = get_mse(rwq_nns, rwq_x_test, rwq_y_test)

     # Generate graph for Red Wine Quality
    plot = Plotter(
        name='Red Wine Quality', 
        learner='neural-network',
        axes={ 'x': 'Number of weight updates', 'y': 'Error' }
    )
    plot.add_plot(rwq_iter_counts, rwq_train_mse, 'training data', 'None')
    plot.add_plot(rwq_iter_counts, rwq_test_mse, 'testing data', 'None')
    plot.save(top_limit=1.)