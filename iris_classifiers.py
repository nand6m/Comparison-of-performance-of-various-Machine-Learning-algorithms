# CS6375 - Machine Learning: Assignment - 5 - Comparison of Classifiers

# data handling libs
import numpy as np
import pandas as pd
import warnings

# Ignore display of unnecessary warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
# data preprocessing libs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# sklearn classifiers to import
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# tensorflow classifier import
import tensorflow as tf
from tensorflow.contrib.learn import DNNClassifier

# model building, predict, accuracy imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from IPython.display import display

# Logging level
tf.logging.set_verbosity(tf.logging.FATAL)

# Get data from csv file 
data = pd.read_csv("iris.csv", names=['sp_length', 'sp_width', 'p_length', 'p_width', 'class'])
print('Dataset used: Iris Data set')
print('Number of instances in dataset:', len(data))
print('Number of attributes in dataset:', len(data.columns.values)-1)
num_folds = 15

# categorize output class labels to numeric values
le = LabelEncoder()
le.fit(data['class'])
data['class'] = le.transform(data['class'])

# Remove any NAN rows from the dataset
data.dropna(inplace=True)

# separate feature data and target data
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)


# Build parameters of all classifiers
random_forest_params = dict(n_estimators=[5, 10, 15, 20, 25], criterion=['gini', 'entropy'], 
                            max_features=[2, 3, 4, 'auto', 'log2', 'sqrt', None], bootstrap=[False, True]
                            )
decision_tree_params = dict(criterion=['gini', 'entropy'], splitter=['best', 'random'], min_samples_split=[2, 3, 4],
                            max_features=[2,3,'auto', 'log2', 'sqrt', None], class_weight=['balanced', None], presort=[False, True])

perceptron_params = dict(penalty=[None, 'l2', 'l1', 'elasticnet'], fit_intercept=[False, True], shuffle=[False, True],
                         class_weight=['balanced', None], alpha=[0.0001, 0.00025], max_iter=[30,50,90])

svm_params = dict(shrinking=[False, True], degree=[3,4], class_weight=['balanced', None])

neural_net_params = dict(activation=['identity', 'logistic', 'tanh', 'relu'], hidden_layer_sizes = [(20,15,10),(30,20,15,10),(16,8,4)], 
                         max_iter=[50,80,150], solver=['adam','lbfgs'], learning_rate=['constant', 'invscaling', 'adaptive'], shuffle=[True, False])

log_reg_params = dict(class_weight=['balanced', None], solver=['newton-cg', 'lbfgs', 'liblinear', 'sag'], fit_intercept=[True, False])

knn_params = dict(n_neighbors=[2, 3, 5, 10], weights=['uniform', 'distance'],
                  algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], leaf_size=[5,10,15,20])

bagging_params = dict(n_estimators=[5, 12, 15, 20], bootstrap=[False, True])

ada_boost_params = dict(n_estimators=[50, 75, 100], algorithm=['SAMME', 'SAMME.R'])

guassiannb_params = dict()

gradient_boosting_params = dict(n_estimators=[15, 25, 50])

params = [
    random_forest_params, decision_tree_params, perceptron_params,
    svm_params, neural_net_params, log_reg_params, knn_params,
    bagging_params, ada_boost_params, guassiannb_params, gradient_boosting_params
]
# classifiers to test
classifiers = [
    RandomForestClassifier(), DecisionTreeClassifier(), Perceptron(),
    SVC(), MLPClassifier(), LogisticRegression(),
    KNeighborsClassifier(), BaggingClassifier(), AdaBoostClassifier(),
    GaussianNB(), GradientBoostingClassifier()
]

names = [
    'RandomForest', 'DecisionTree', 'Perceptron', 'SVM',
    'NeuralNetwork', 'LogisticRegression',
    'KNearestNeighbors', 'Bagging', 'AdaBoost', 'Naive-Bayes', 'GradientBoosting'
]

models = dict(zip(names, zip(classifiers, params)))

#Finding best parameters using Gridsearch 
def parameter_tuning(models, X_train, X_test, y_train, y_test):
    print(num_folds,'fold cross-validation is used')
    print()
    accuracies = []
    # dataframe to store intermediate results
    dataframes = []
    best_parameters = []
    for name, clf_and_params in models.items():
        print('Computing GridSearch on {} '.format(name))
        clf, clf_params = clf_and_params
        grid_clf = GridSearchCV(estimator=clf, param_grid=clf_params, cv=num_folds)
        grid_clf = grid_clf.fit(X_train, y_train)
        dataframes.append((name, grid_clf.cv_results_))
        best_parameters.append((name, grid_clf.best_params_))
        predictions = grid_clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=num_folds)
        accuracies.append((name, accuracy, np.mean(cv_scores)))
    return accuracies, dataframes, best_parameters


results, dataframes, best_parameters = parameter_tuning(models, X_train, X_test, y_train, y_test)
print()
print('============================================================')
for classifier, acc, cv_acc in results:
    print('{}: Accuracy with Best Parameters = {}% || Mean Cross Validation Accuracy = {}%'.format(classifier, round(acc*100,4), round(cv_acc*100,4)))
print()

for name, bp in best_parameters:
    print('============================================================')
    print('{} classifier GridSearch Best Parameters'.format(name))
    display(bp)
print()
print()

# Deep Learning using Tensor flow
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(X[0]))]
deep_learning = DNNClassifier(hidden_units=[10,20,10],
                      feature_columns=feature_columns, model_dir="/tmp/iris")
deep_learning.fit(X_train, y_train, steps=1500)
predictions = list(deep_learning.predict(X_test, as_iterable=True))
acc = accuracy_score(predictions, predictions)
print('============================================================')
print('Deep Learning classifier Accuracy = ', round(acc*100,4),'%')
print('------------------------------------------------------------')
print('Deep Learning classifier Best Parameters')
display(deep_learning.params)
print('***************** Execution Completed **********************')
print('------------------------------------------------------------')
