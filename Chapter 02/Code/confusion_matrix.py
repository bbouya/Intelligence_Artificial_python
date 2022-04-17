# Importation de library 
from xml.etree.ElementPath import prepare_descendant
import numpy as np

import matplotlib.pyplot as plt
from pyparsing import condition_as_parse_action

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define sample labels :
true_labels = [2,0,0,2,4,4,1,0,3,3,3]

pred_label = [2,1,0,2,4,3,1,0,1,3,3]

# Create confusion matrix 
confusion_mat = confusion_matrix(true_labels,pred_label)

# Visualize confusion matrix 

plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion Matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks,ticks)
plt.ylabel("True Labels")
plt.xlabel('Prediction labels')
plt.show()


# Classification report :
targets = ['Class-0', 'Class-1','Class-2', 'Class-3','Class-4']
print('\n', classification_report(true_labels, target_name = targets))
