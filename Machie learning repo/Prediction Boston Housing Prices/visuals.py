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