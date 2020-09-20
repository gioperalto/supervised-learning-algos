import numpy as np
from utils import red_wine_quality, heart_failure_prediction, get_mse, scale_data, benchmarks
from plotter import Plotter
from sklearn.svm import SVC

def create_svm(x, y, kernel='linear', gamma=0.001):
    return SVC(
        kernel=kernel, 
        gamma=gamma,
        random_state=0
    ).fit(x, y.values.flatten())

def create_rbf_svm(x, y, gamma=0.001):
    return SVC(
        kernel='rbf',
        gamma=gamma,
        random_state=0
    ).fit(x, y.values.flatten())

def create_svms(x, y, kernel, gammas):
    boosts = []

    for g in gammas:
        boosts.append(create_svm(x, y, kernel, g))

    return boosts

if __name__ == "__main__":
    # Split testing/training data and create trees
    hfp_x_train, hfp_y_train, hfp_x_test, hfp_y_test = heart_failure_prediction()
    rwq_x_train, rwq_y_train, rwq_x_test, rwq_y_test = red_wine_quality()

    hfp_size, rwq_size = hfp_x_train.shape[0], rwq_x_train.shape[0]
    hfp_gammas = np.arange(1./hfp_size, 1., .01)
    rwq_gammas = np.arange(1./rwq_size, 1., .01)

    # Scale data
    hfp_x_train, hfp_x_test = scale_data(hfp_x_train, hfp_x_test)
    rwq_x_train, rwq_x_test = scale_data(rwq_x_train, rwq_x_test)

    # Create Heart Failure Prediction SVMs
    hfp_poly_svms = create_svms(hfp_x_train, hfp_y_train, 'poly', hfp_gammas)
    hfp_rbf_svms = create_svms(hfp_x_train, hfp_y_train, 'rbf', hfp_gammas)
    hfp_sigmoid_svms = create_svms(hfp_x_train, hfp_y_train, 'sigmoid', hfp_gammas)

    # Get training/testing accuracy for Heart Failure Prediction
    hfp_poly_train_mse = get_mse(hfp_poly_svms, hfp_x_train, hfp_y_train)
    hfp_poly_test_mse = get_mse(hfp_poly_svms, hfp_x_test, hfp_y_test)
    hfp_rbf_train_mse = get_mse(hfp_rbf_svms, hfp_x_train, hfp_y_train)
    hfp_rbf_test_mse = get_mse(hfp_rbf_svms, hfp_x_test, hfp_y_test)
    hfp_sigmoid_train_mse = get_mse(hfp_sigmoid_svms, hfp_x_train, hfp_y_train)
    hfp_sigmoid_test_mse = get_mse(hfp_sigmoid_svms, hfp_x_test, hfp_y_test)

    print('\nHeart Failure Prediction - Benchmarks (SVM):')
    benchmarks(hfp_x_train, hfp_y_train, hfp_x_test, hfp_y_test, hfp_gammas, hfp_rbf_test_mse, create_rbf_svm)

    # Generate graph for Heart Failure Prediction
    plot = Plotter(
        name='Heart Failure Prediction', 
        learner='svm', 
        axes={ 'x': 'γ', 'y': 'Error' }
    )
    plot.add_plot(hfp_gammas, hfp_poly_train_mse, 'training data (poly)', 'None')
    plot.add_plot(hfp_gammas, hfp_rbf_train_mse, 'training data (rbf)', 'None')
    plot.add_plot(hfp_gammas, hfp_sigmoid_train_mse, 'training data (sigmoid)', 'None')
    plot.add_plot(hfp_gammas, hfp_poly_test_mse, 'testing data (poly)', 'None')
    plot.add_plot(hfp_gammas, hfp_rbf_test_mse, 'testing data (rbf)', 'None')
    plot.add_plot(hfp_gammas, hfp_sigmoid_test_mse, 'testing data (sigmoid)', 'None')
    plot.save(loc='lower right', framealpha=.25)

    # Create Red Wine Quality SVMs
    rwq_poly_svms = create_svms(rwq_x_train, rwq_y_train, 'poly', rwq_gammas)
    rwq_rbf_svms = create_svms(rwq_x_train, rwq_y_train, 'rbf', rwq_gammas)
    rwq_sigmoid_svms = create_svms(rwq_x_train, rwq_y_train, 'sigmoid', rwq_gammas)

    # Get training/testing accuracy for Red Wine Quality
    rwq_poly_train_mse = get_mse(rwq_poly_svms, rwq_x_train, rwq_y_train)
    rwq_poly_test_mse = get_mse(rwq_poly_svms, rwq_x_test, rwq_y_test)
    rwq_rbf_train_mse = get_mse(rwq_rbf_svms, rwq_x_train, rwq_y_train)
    rwq_rbf_test_mse = get_mse(rwq_rbf_svms, rwq_x_test, rwq_y_test)
    rwq_sigmoid_train_mse = get_mse(rwq_sigmoid_svms, rwq_x_train, rwq_y_train)
    rwq_sigmoid_test_mse = get_mse(rwq_sigmoid_svms, rwq_x_test, rwq_y_test)

    print('\nRed Wine Quality - Benchmarks (SVM):')
    benchmarks(rwq_x_train, rwq_y_train, rwq_x_test, rwq_y_test, rwq_gammas, rwq_rbf_test_mse, create_rbf_svm)


    # Generate graph for Red Wine Quality 
    plot = Plotter(
        name='Red Wine Quality', 
        learner='svm', 
        axes={ 'x': 'γ', 'y': 'Error' }
    )
    plot.add_plot(rwq_gammas, rwq_poly_train_mse, 'training data (poly)', 'None')
    plot.add_plot(rwq_gammas, rwq_rbf_train_mse, 'training data (rbf)', 'None')
    plot.add_plot(rwq_gammas, rwq_sigmoid_train_mse, 'training data (sigmoid)', 'None')
    plot.add_plot(rwq_gammas, rwq_poly_test_mse, 'testing data (poly)', 'None')
    plot.add_plot(rwq_gammas, rwq_rbf_test_mse, 'testing data (rbf)', 'None')
    plot.add_plot(rwq_gammas, rwq_sigmoid_test_mse, 'testing data (sigmoid)', 'None')
    plot.save(loc='lower left', framealpha=.25)