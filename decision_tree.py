from utils import red_wine_quality, heart_failure_prediction, get_mse
from plotter import Plotter
from sklearn.tree import DecisionTreeClassifier, export_text

def print_tree(tree):
    print(export_text(tree))

def create_trees(x, y):
    dts = []

    # Create initial tree
    dt = DecisionTreeClassifier(random_state=0)
    dt = dt.fit(x, y)
    path = dt.cost_complexity_pruning_path(x, y)

    # Create trees (and prune more as alpha increases)
    for alpha in path.ccp_alphas:
        dt = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
        dt.fit(x, y)
        dts.append(dt)
    node_counts = [dt.tree_.node_count for dt in dts]

    return dts, path.ccp_alphas, node_counts

if __name__ == "__main__":
    # Split testing/training data and create trees
    hfp_x_train, hfp_y_train, hfp_x_test, hfp_y_test = heart_failure_prediction()
    rwq_x_train, rwq_y_train, rwq_x_test, rwq_y_test = red_wine_quality()
    hfp_trees, hfp_alphas, hfp_node_counts = create_trees(hfp_x_train, hfp_y_train)
    rwq_trees, rwq_alphas, rwq_node_counts = create_trees(rwq_x_train, rwq_y_train)

    # Get training/testing accuracy for Red Wine Quality + Heart Failure Prediction
    rwq_train_mse = get_mse(rwq_trees, rwq_x_train, rwq_y_train)
    rwq_test_mse = get_mse(rwq_trees, rwq_x_test, rwq_y_test)
    hfp_train_mse = get_mse(hfp_trees, hfp_x_train, hfp_y_train)
    hfp_test_mse = get_mse(hfp_trees, hfp_x_test, hfp_y_test)

    # Generate graph for Red Wine Quality
    plot = Plotter('Red Wine Quality', 'decision-tree', { 'x': 'α', 'y': 'Error' })
    plot.add_plot(rwq_alphas, rwq_train_mse, 'training data', 'None')
    plot.add_plot(rwq_alphas, rwq_test_mse, 'testing data', 'None')
    plot.find_min(rwq_alphas, rwq_test_mse, 'testing')
    plot.save()

    # Generate graph for Heart Failure Prediction
    plot = Plotter('Heart Failure Prediction', 'decision-tree', { 'x': 'α', 'y': 'Error' })
    plot.add_plot(hfp_alphas, hfp_train_mse, 'training data', 'None')
    plot.add_plot(hfp_alphas, hfp_test_mse, 'testing data', 'None')
    plot.find_min(hfp_alphas, hfp_test_mse, 'testing')
    plot.save()