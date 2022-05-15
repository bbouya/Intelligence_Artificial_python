import warnings 
warnings.filterwarnings('ignore', category=UserWarning, module = 'matplotlib')

# Display inline matplotlib plots with Ipython

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as curves
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split

def ModelLearning(X, y):
    """
        Calculates the performance of several model with varyin sizes of training data.
        The learning and testing scores for each model are then plotted
    """
    # Create 10 cross-validation sets for training and testing
    # Cross-validation is a resampling procedure used to evaluate machine leaning model on a limited data sample, the procedure has a single parameter called K that refers to the numer of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation

    cv = ShuffleSplit(X.shape[0], test_size = 0.2, random_state = 0)
    
    # Gnerate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window : 
    fig = plt.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, depth in  enumerate([1,3,6,10]):

        # Create a decision tree regressor at max_depth = depth
        regressor = DecisionTreeRegressor(max_depth= depth)

        # Calculate the training and testing scores 
        size, train_scores, test_scores = curves.learning_curve(regressor, X, y, \
            cv = cv, train_sizes = train_sizes, scoring= 'r2')

        # Find the mean and standard deviation for smoothing 
        train_std = np.std(train_scores, axis = 1)
        
