import pydot as py
from IPython.display import SVG
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import model_to_dot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc, mean_absolute_error
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
def create_model(dense_layer_n, last_layer_activation):
    model = Sequential()
    model.add(Dense(dense_layer_n, activation=last_layer_activation, input_dim=len(elo_data.columns) - 1))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=last_layer_activation))

    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def convertSpecificColumnToNumeric(value, dataset):
    for column in dataset.columns:
        if column == value:
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
            print("Converting '{0}' variable to an integer".format(column))


# Variable Elo
elo_data = pd.read_csv('./Elo_Variable_HomeAdv2.csv')

# get dimensions
print(elo_data.shape)

print(elo_data.head())

print(elo_data.columns)

# descriptive stats
print(elo_data.describe())

elo_data_grouped = elo_data.groupby(['Season', 'Curr_WinLoss'])

print(elo_data_grouped['Curr_WinLoss'].describe().unstack())

Y_data = elo_data['Curr_WinLoss']

encoder = LabelBinarizer()
y_cat = encoder.fit_transform(Y_data)
nlabels = len(encoder.classes_)

X_data = elo_data.drop(['Curr_WinLoss'], axis=1)

for column in X_data.columns:
    if X_data[column].dtype == type(object):
        convertSpecificColumnToNumeric(column, X_data)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

random.seed(100)
X_train_elo, X_test_elo, Y_train_elo, Y_test_elo = train_test_split(X_data, y_cat, test_size=0.30)

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

param_grid = dict(batch_size=batch_size, epochs=epochs, dense_layer_n=layer_size,last_layer_activation=last_layer_activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train_elo, Y_train_elo)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#
#
#   RUN MODEL
#
#

model = Sequential()

model.add(Dense(20, activation='relu', input_dim=len(elo_data.columns) - 1))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_elo, Y_train_elo, epochs=10, batch_size=50, validation_split=0.2, verbose=1)
model.test_on_batch(X_test_elo, Y_test_elo, sample_weight=None)
model.evaluate(X_test_elo, Y_test_elo, verbose=1)
pred = model.predict_classes(X_test_elo, verbose=1)

plot_model(model, to_file='model.png', show_shapes=True)

SVG(model_to_dot(model).create(prog='dot', format='svg'))

print(confusion_matrix(Y_test_elo, pred))
print classification_report(Y_test_elo, pred)
print(accuracy_score(Y_test_elo, pred))
fpr_elo, tpr_elo, thresholds_elo = roc_curve(Y_test_elo, pred)

auc = auc(fpr_elo, tpr_elo)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_elo, tpr_elo, label='Rating System Model (AUC = {:.3f})'.format(auc))
# plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
