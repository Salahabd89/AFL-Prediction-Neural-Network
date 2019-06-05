import pydot as py
from IPython.display import SVG
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import roc_curve, auc, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense


# Function Definitions
# Function Definitions
def create_model(dense_layer_n, optimizer):
    model = Sequential()
    model.add(Dense(dense_layer_n, activation='relu', input_dim=X_data.shape[1]))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def convertSpecificColumnToNumeric(value, dataset):
    for column in dataset.columns:
        if column == value:
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
            print("Converting '{0}' variable to an integer".format(column))


# Performance Based
perf_data = pd.read_csv('./Performance_based_Model_2_Data.csv')

# get dimensions
print(perf_data.shape)

print(perf_data.head())

print(perf_data.columns)

# descriptive stats
print(perf_data.describe())

Y_data = perf_data['Curr_WinLoss']

encoder = LabelBinarizer()
y_cat = encoder.fit_transform(Y_data)
nlabels = len(encoder.classes_)

X_data = perf_data.drop(['Curr_WinLoss', 'Unnamed: 0'], axis=1)

for column in X_data.columns:
    if X_data[column].dtype == type(object):
        convertSpecificColumnToNumeric(column, X_data)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
random.seed(100)
X_train_Perf_based, X_test_Perf_based, Y_train_Perf_based, Y_test_Perf_based = train_test_split(X_data, y_cat, test_size=0.30)

#
#
#   GRID SEARCH
#
#

model = KerasClassifier(build_fn=create_model, verbose=0)

layer_size = [20, 40, 60, 80, 100]
batch_size = [50, 100]
epochs = [10, 50, 100]
last_layer_activation = ['sigmoid','relu']
optimizer = ['sgd','adam']

param_grid = dict(batch_size=batch_size, optimizer=optimizer, epochs=epochs, dense_layer_n=layer_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X_train_Perf_based, Y_train_Perf_based)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#
#
#   Build the Model
#
#

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=X_data.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_Perf_based, Y_train_Perf_based, epochs=100, batch_size=20, validation_split=0.3, verbose=1)

model.test_on_batch(X_test_Perf_based, Y_test_Perf_based, sample_weight=None)

pred_Perf_based = model.predict_classes(X_test_Perf_based, verbose=1)

print(confusion_matrix(Y_test_Perf_based, pred_Perf_based))
print classification_report(Y_test_Perf_based, pred_Perf_based)
print(accuracy_score(Y_test_Perf_based, pred_Perf_based))
fpr_Perf_based, tpr_Perf_based, thresholds_Perf_based = roc_curve(Y_test_Perf_based, pred_Perf_based)

auc_keras = auc(fpr_Perf_based, tpr_Perf_based)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_Perf_based, tpr_Perf_based, label='Performance Based Model (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
