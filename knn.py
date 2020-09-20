import numpy as np
from plotter import Plotter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from utils import red_wine_quality, heart_failure_prediction, get_mse, scale_data, benchmarks

def create_knn(x, y, n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors).fit(x, y.values.flatten())

def create_knns(x, y, neighbors):
    knns = []

    for n in neighbors:
        knns.append(create_knn(x, y, n))

    return knns

def generate_graphs_hfp(hfp_neighbors, hfp_train_mse, hfp_test_mse):
    # Generate graph for Heart Failure Prediction
    plot = Plotter(
        name='Heart Failure Prediction', 
        learner='knn', 
        axes={ 'x': 'n_neighbors', 'y': 'Error' }
    )
    plot.add_plot(hfp_neighbors, hfp_train_mse, 'training data', 'None')
    plot.add_plot(hfp_neighbors, hfp_test_mse, 'testing data', 'None')
    plot.find_min_int(hfp_neighbors, hfp_test_mse, 'testing', top=False)
    plot.save()

def generate_graphs_rwq(rwq_neighbors, rwq_train_mse, rwq_test_mse):
    # Generate graph for Red Wine Quality
    plot = Plotter(
        name='Red Wine Quality', 
        learner='knn', 
        axes={ 'x': 'n_neighbors', 'y': 'Error' }
    )
    plot.add_plot(rwq_neighbors, rwq_train_mse, 'training data', 'None')
    plot.add_plot(rwq_neighbors, rwq_test_mse, 'testing data', 'None')
    plot.find_min_int(rwq_neighbors, rwq_test_mse, 'testing')
    plot.save()

if __name__ == "__main__":
    # Split testing/training data and create trees
    hfp_x_train, hfp_y_train, hfp_x_test, hfp_y_test = heart_failure_prediction()
    rwq_x_train, rwq_y_train, rwq_x_test, rwq_y_test = red_wine_quality()

    # Scale data
    hfp_x_train, hfp_x_test = scale_data(hfp_x_train, hfp_x_test)
    rwq_x_train, rwq_x_test = scale_data(rwq_x_train, rwq_x_test)

    hfp_neighbors = np.arange(1, hfp_x_train.shape[0]+1, 1, dtype=int)
    rwq_neighbors = np.arange(1, rwq_x_train.shape[0]+1, 5, dtype=int)

    # Create Heaart Failure Prediction KNNs
    hfp_knns = create_knns(hfp_x_train, hfp_y_train, hfp_neighbors)

    # Get training/testing accuracy for Heart Failure Prediction
    hfp_train_mse = get_mse(hfp_knns, hfp_x_train, hfp_y_train)
    hfp_test_mse = get_mse(hfp_knns, hfp_x_test, hfp_y_test)

    generate_graphs_hfp(hfp_neighbors, hfp_train_mse, hfp_test_mse)
    print('\nHeart Failure Prediction - Benchmarks (KNN):')
    benchmarks(hfp_x_train, hfp_y_train, hfp_x_test, hfp_y_test, hfp_neighbors, hfp_test_mse, create_knn)

    # # Create Red Wine Quality KNNs
    rwq_knns = create_knns(rwq_x_train, rwq_y_train, rwq_neighbors)

    # Get training/testing accuracy for Red Wine Quality
    rwq_train_mse = get_mse(rwq_knns, rwq_x_train, rwq_y_train)
    rwq_test_mse = get_mse(rwq_knns, rwq_x_test, rwq_y_test)

    generate_graphs_rwq(rwq_neighbors, rwq_train_mse, rwq_test_mse)
    print('\nRed Wine Quality - Benchmarks (KNN):')
    benchmarks(rwq_x_train, rwq_y_train, rwq_x_test, rwq_y_test, rwq_neighbors, rwq_test_mse, create_knn)