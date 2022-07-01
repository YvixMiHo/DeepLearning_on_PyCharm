# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#Import dependencies
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, losses
import matplotlib.pyplot as plt

#Load data into enviroment
(train_imgs, train_labels),(test_imgs, test_labels) = datasets.cifar10.load_data()

#Process the data
train_imgs, test_imgs = train_imgs/255.0, test_imgs/255.0

#Validate data shape
print(f"Training images has shape: {train_imgs.shape} and dtype: {train_imgs.dtype}")
print(f"Training labels has shape: {train_labels.shape} and dtype: {train_labels.dtype}")
print(f"Validation images has shape: {test_imgs.shape} and dtype: {test_imgs.dtype}")
print(f"Validation labels has shape: {test_labels.shape} and dtype: {test_labels.dtype}")

#visualize the data
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print("classes shape is: ", len(classes))

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_imgs[i])
    plt.xlabel(classes[train_labels[i][0]])
plt.show()

#Model,compile, fit
model = models.Sequential()
#input layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())

#output layer
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#compile
model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

history = model.fit(train_imgs, train_labels,
                    epochs=10, validation_split=0.2)

#plot loss vs accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')