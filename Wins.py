import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.cross_validation import cross_val_score

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# Function Definitions

def convertSpecificColumnToNumeric(value, dataset):
    for column in dataset.columns:
        if column == value:
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
            print( "Converting '{0}' variable to an integer".format(column))

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d_t-%d' % (j + 1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d_t' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d_t+%d' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


dataset = pd.read_csv('/Users/salahabdallah/Downloads/aflstats/timed.csv')

values = dataset.values

reframed = series_to_supervised(values, 1, 1)

#reframed.to_csv('//Users/salahabdallah/Desktop/Uni/Year 2/Semester 1/Sports/final/reframed.csv')


#reframed = pd.read_csv('/Users/salahabdallah/Desktop/Uni/Year 2/Semester 1/Sports/final/reframed2.csv')

reframed.rename(index={0:'zero',1:'one'}, inplace=True)

Y_data = reframed['WinLoss']

encoder = LabelBinarizer()
y_cat = encoder.fit_transform(Y_data)
nlabels = len(encoder.classes_)

X_data = reframed.drop(['WinLoss'], axis=1)

for column in X_data.columns:
    if X_data[column].dtype == type(object):
        convertSpecificColumnToNumeric(column,X_data)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#scaler =StandardScaler()
#X_data = scaler.fit_transform(X_data)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_cat, test_size=0.30)

model = Sequential()
model.add(Dense(26, activation='relu', input_dim=26))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train.values, Y_train, epochs=1000, batch_size=10, validation_split=0.2, verbose=1)
model.test_on_batch(X_test.values, Y_test, sample_weight=None)










