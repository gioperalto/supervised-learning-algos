import numpy as np
from utils import red_wine_quality, heart_failure_prediction, get_mse, scale_data
from plotter import Plotter
from sklearn.ensemble import AdaBoostClassifier

def create_adaboost(x, y, n_estimators=100, lr=1.):
    return AdaBoostClassifier(
        n_estimators=n_estimators, 
        learning_rate=lr,
        random_state=0
    ).fit(x, y.values.flatten())

def create_adaboosts(x, y, n_estimators=100, lr=1.):
    boosts = []

    for i in n_estimators:
        boosts.append(create_adaboost(x, y, i, lr))

    return boosts

if __name__ == "__main__":
    # Split testing/training data and create trees
    hfp_x_train, hfp_y_train, hfp_x_test, hfp_y_test = heart_failure_prediction()
    rwq_x_train, rwq_y_train, rwq_x_test, rwq_y_test = red_wine_quality()

    n_estimators_set = np.arange(1,101, 1, dtype=int)

    # Scale data
    hfp_x_train, hfp_x_test = scale_data(hfp_x_train, hfp_x_test)
    rwq_x_train, rwq_x_test = scale_data(rwq_x_train, rwq_x_test)

    # Create Heart Failure Prediction NNs
    hfp_adas_l2 = create_adaboosts(hfp_x_train, hfp_y_train, n_estimators_set, .1)

    # Get training/testing accuracy for Heart Failure Prediction
    hfp_train_mse_l2 = get_mse(hfp_adas_l2, hfp_x_train, hfp_y_train)
    hfp_test_mse_l2 = get_mse(hfp_adas_l2, hfp_x_test, hfp_y_test)

    # Generate graph for Heart Failure Prediction
    plot = Plotter(
        name='Heart Failure Prediction', 
        learner='adaboost', 
        axes={ 'x': 'Number of estimators', 'y': 'Error' }
    )
    plot.add_plot(n_estimators_set, hfp_train_mse_l2, 'training data (lr=0.1)', 'None')
    plot.add_plot(n_estimators_set, hfp_test_mse_l2, 'testing data (lr=0.1)', 'None')
    plot.save()

    # Create Heart Failure Prediction NNs
    rwq_adas_l2 = create_adaboosts(rwq_x_train, rwq_y_train, n_estimators_set, .1)

    # Get training/testing accuracy for Heart Failure Prediction
    rwq_train_mse_l2 = get_mse(rwq_adas_l2, rwq_x_train, rwq_y_train)
    rwq_test_mse_l2 = get_mse(rwq_adas_l2, rwq_x_test, rwq_y_test)

    # Generate graph for Red Wine Quality
    plot = Plotter(
        name='Red Wine Quality', 
        learner='adaboost', 
        axes={ 'x': 'Number of estimators', 'y': 'Error' }
    )
    plot.add_plot(n_estimators_set, rwq_train_mse_l2, 'training data (lr=0.1)', 'None')
    plot.add_plot(n_estimators_set, rwq_test_mse_l2, 'testing data (lr=0.1)', 'None')
    plot.save()