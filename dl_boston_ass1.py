import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import seaborn as sns
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'Price' ]
house_df =  pd.read_csv(url, sep= '\s+', names=col_names )
# house_df.head(),columns,shape, isnul().sum(), describe()
corr = house_df.corr()
plt.figure(figsize = (20,20))
sns.heatmap(corr, cbar = True , square = True , fmt = '1f' , annot = True , annot_kws = {'size':15},cmap='YlGnBu')
feature = house_df.iloc[:,0:13] 
target = house_df.iloc[:,13]
print(feature.head())
print('\n',target.head())
normalized_feature =  keras.utils.normalize(feature.values)
print(normalized_feature)
X_train, X_test, y_train, y_test = train_test_split(normalized_feature, target.values,test_size=0.2, random_state=42) 
print('training data shape: ',X_train.shape)
print('testing data shape: ',X_test.shape)
n_cols = X_train.shape[1]
model = keras.Sequential()
model.add(keras.layers.Dense(150, activation=tf.nn.relu,input_shape=(n_cols,)))
model.add(keras.layers.Dense(150, activation=tf.nn.relu))# 3 more
model.add(keras.layers.Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=15) 
history = model.fit(X_train, y_train, epochs=300,validation_split=0.2, verbose=1, callbacks=[early_stop])
plt.figure(figsize=(15,8))
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.plot(history.epoch, history.history['loss'],label='Train Loss')
plt.plot(history.epoch, history.history['val_loss'],label = 'Val loss')
plt.title('Model loss')
plt.legend()
score = model.evaluate(X_test, y_test, verbose=1)
print('loss value: ', score[0])
print('Mean absolute error: ', score[1])
test_predictions = model.predict(X_test).flatten()
print(test_predictions)
true_predicted = pd.DataFrame(list(zip(y_test, test_predictions)),columns=['True Value','Predicted Value'])
true_predicted.head(10)
x = test_predictions
y = y_test
plt.figure(figsize=(30,10))
plt.plot(x, label='predicted value')
plt.plot(y, label='true value')
plt.title('Evaluation Result')
plt.legend()
plt.show()
y = test_predictions 
x = y_test
fig, ax = plt.subplots(figsize=(10,6)) 
ax.scatter(x,y) 
ax.set(xlim=(0,55), ylim=(0, 55)) 
ax.plot(ax.get_xlim(), ax.get_ylim(), color ='red') 
plt.xlabel('True Values')
plt.ylabel('Predicted values')
plt.title('Evaluation Result')
plt.show()