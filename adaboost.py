import numpy as np
from utils import red_wine_quality, heart_failure_prediction, get_accuracy
from plotter import Plotter
from sklearn.ensemble import AdaBoostClassifier

def create_adaboost(x, y, n_estimators=100):
    return AdaBoostClassifier(
        n_estimators=n_estimators, 
        random_state=0
    ).fit(x, y.values.flatten())

def create_adaboosts(x, y, n_estimators=100):
    boosts = []

    for i in n_estimators:
        boosts.append(create_adaboost(x, y, i))

    return boosts

if __name__ == "__main__":
    # Split testing/training data and create trees
    hfp_x_train, hfp_y_train, hfp_x_test, hfp_y_test = heart_failure_prediction()
    rwq_x_train, rwq_y_train, rwq_x_test, rwq_y_test = red_wine_quality()

    n_estimators_set = np.arange(1,101, 1, dtype=int)

    # Create Heart Failure Prediction NNs
    hfp_adas = create_adaboosts(hfp_x_train, hfp_y_train, n_estimators_set)

    # Get training/testing accuracy for Heart Failure Prediction
    hfp_train_acc = get_accuracy(hfp_adas, hfp_x_train, hfp_y_train)
    hfp_test_acc = get_accuracy(hfp_adas, hfp_x_test, hfp_y_test)

    # Generate graph for Heart Failure Prediction
    plot = Plotter(
        name='Heart Failure Prediction', 
        learner='adaboost', 
        axes={ 'x': 'Number of estimators', 'y': 'Accuracy (%)' }
    )
    plot.add_plot(n_estimators_set, hfp_train_acc, 'training data', 'None')
    plot.add_plot(n_estimators_set, hfp_test_acc, 'testing data', 'None')
    plot.save()

    # Create Red Wine Quality NNs
    rwq_adas = create_adaboosts(rwq_x_train, rwq_y_train, n_estimators_set)

    # Get training/testing accuracy for Red Wine Quality
    rwq_train_acc = get_accuracy(rwq_adas, rwq_x_train, rwq_y_train)
    rwq_test_acc = get_accuracy(rwq_adas, rwq_x_test, rwq_y_test)

     # Generate graph for Red Wine Quality
    plot = Plotter(
        name='Red Wine Quality', 
        learner='adaboost', 
        axes={ 'x': 'Number of estimators', 'y': 'Accuracy (%)' }
    )
    plot.add_plot(n_estimators_set, rwq_train_acc, 'training data', 'None')
    plot.add_plot(n_estimators_set, rwq_test_acc, 'testing data', 'None')
    plot.save()