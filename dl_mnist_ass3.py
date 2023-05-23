import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print shape x y train test
plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()
index = 0
plt.figure(figsize=(5,5))
plt.imshow(x_train[index], cmap=plt.cm.binary)
plt.xlabel(class_names[y_train[index]])
plt.colorbar()
plt.show()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
        filters=32,  
        kernel_size=(3, 3),strides=(1, 1), 
        padding='valid', activation='relu', 
        input_shape=(28, 28, 1))) 
model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),strides=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, 
        activation='relu'))
model.add(tf.keras.layers.Dense(
        units=10, activation='softmax'))
model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy, 
    optimizer=tf.keras.optimizers.Adam(), 
    metrics=['accuracy'] )
model.summary()
tf.keras.utils.plot_model(model, to_file='model.png', 
    show_shapes=True, show_layer_names=False)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
history = model.fit(
      x_train, y_train,
      batch_size=256,epochs=10, 
      validation_split=0.2,verbose=1)
predicted_classes = model.predict(x_test)
y_pred = [np.argmax(predicted_classes[i]) for i in range(len(predicted_classes))]
print(classification_report(y_test, y_pred, target_names=class_names))
correct = np.nonzero(y_pred==y_test)[0]
plt.figure(figsize=(15, 8))
for j, correct in enumerate(correct[0:8]):
    plt.subplot(2, 4, j+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[correct].reshape(28, 28), cmap="Reds")
    plt.title("Predicted: {}".format(class_names[y_pred[correct]]))
    plt.xlabel("Actual: {}".format(class_names[y_test[correct]]))