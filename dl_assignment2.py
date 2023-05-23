import tensorflow as tf
import os,re,string
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
raw_data_df = pd.read_csv('dataset/IMDB_Dataset.csv')
raw_data_df.head()
raw_data_df['sentiment'] = raw_data_df['sentiment'].apply(lambda row : 1 if row == 'positive' else 0)
raw_data_df.head()
len(raw_data_df[raw_data_df['sentiment'] == 1])
len(raw_data_df[raw_data_df['sentiment'] == 0])
features = raw_data_df['review'].to_numpy()
labels = raw_data_df['sentiment'].to_numpy()
features.shape,labels.shape
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.4, random_state = 0)
features_valid, features_test, labels_valid, labels_test = train_test_split(features_test, labels_test, test_size=0.5, random_state=0)
features_train = tf.convert_to_tensor(features_train)
labels_train = tf.convert_to_tensor(labels_train)
features_valid = tf.convert_to_tensor(features_valid)
labels_valid = tf.convert_to_tensor(labels_valid)
features_test = tf.convert_to_tensor(features_test)
labels_test = tf.convert_to_tensor(labels_test)
train_ds = tf.data.Dataset.from_tensor_slices((features_train, labels_train))
next(iter(train_ds))
valid_ds = tf.data.Dataset.from_tensor_slices((features_valid, labels_valid))
next(iter(valid_ds))
test_ds = tf.data.Dataset.from_tensor_slices((features_test, labels_test))
next(iter(test_ds))
BATCH_SIZE = 64
train_ds = train_ds.batch(batch_size=BATCH_SIZE)
train_ds.cardinality()
valid_ds = valid_ds.batch(batch_size=BATCH_SIZE)
train_ds.cardinality()
test_ds = test_ds.batch(batch_size=BATCH_SIZE)
test_ds.cardinality()
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
def custom_standardization(input_data):
  std_text = tf.strings.lower(input_data)#remove any urls from the text
  std_text = tf.strings.regex_replace(std_text, r"https:\/\/.*[\r\n]*", '')
  std_text = tf.strings.regex_replace(std_text, r"www\.\w*\.\w\w\w", '')
  std_text = tf.strings.regex_replace(std_text, r"<[\w]*[\s]*/>", '')
  std_text = tf.strings.regex_replace(std_text, '[%s]' % re.escape(string.punctuation), '')
  std_text = tf.strings.regex_replace(std_text, '\s{2}', '')
  std_text = tf.strings.strip(std_text)
  return std_text
custom_standardization("Hello ! <br /> I am here. Why are you upset ?").numpy()
VOCAB_SIZE = 1000
vectorizer_layer = tf.keras.layers.experimental.
                    preprocessing.TextVectorization(max_tokens=VOCAB_SIZE,
                    standardize=custom_standardization)
vectorizer_layer.adapt(train_ds.map(lambda text, label: text))
vocab = np.array(vectorizer_layer.get_vocabulary())
examples, labels = next(iter(train_ds.take(1)))
vectorized_examples = vectorizer_layer(examples)
vectorized_examples.shape
model = tf.keras.Sequential([ vectorizer_layer,
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1, ) ])
sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),metrics=['accuracy'])
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)
history = model.fit(train_ds,
  epochs=10,validation_data=valid_ds)
test_loss, test_acc = model.evaluate(test_ds)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
import matplotlib.pyplot as plt
plt.plot(history.history["loss"], "x-", label="Train_Loss")
plt.plot(history.history["val_loss"], "x-", label="Valid_Loss")
plt.plot(history.history["accuracy"], "x-", label="Train_Accuracy")
plt.plot(history.history["val_accuracy"], "x-", label="Valid_Accuracy")
plt.title("Train And Valid Loss & Accuracy")
plt.legend(loc="best")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()