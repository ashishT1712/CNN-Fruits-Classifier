# Convolutional Neural Network

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



no_of_classes = 60

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (5, 5), input_shape = (100, 100, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2), strides= 2))


# Adding a second convolutional layer
classifier.add(Conv2D(64, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides= 2))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides= 2))

# Adding a third convolutional layer
classifier.add(Conv2D(256, (5, 5), activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides= 2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = no_of_classes, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

Training = train_datagen.flow_from_directory('Fruit-Images-Dataset/Training',
                                                 target_size = (100, 100),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

Validation = test_datagen.flow_from_directory('Fruit-Images-Dataset/Validation',
                                            target_size = (100, 100),
                                            batch_size = 32,
                                            class_mode = 'categorical')


history = classifier.fit_generator(Training,
                         steps_per_epoch = 4000,
                         epochs = 25,
                         validation_data = Validation,
                         validation_steps = 4000)
classifier.summary()


#Plotting graphs
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()



name_of_classes = [ "Apple Braeburn", "Apple Golden 1", "Apple Golden 2", "Apple Golden 3" , "Apple Granny Smith" ,"Apple Red 1" ,"Apple Red 2" ,"Apple Red 3" ,"Apple Red Delicious" ,"Apple Red Yellow" ,"Apricot" ,"Avocado","Avocado ripe" ,"Banana" ,"Banana Red" ,"Cactus fruit" ,"Carambula" ,"Cherry" ,"Clementine" ,"Cocos" ,"Dates" ,"Granadilla" ,"Grape Pink" ,"Grape White" ,"Grape White 2" ,"Grapefruit Pink" ,"Grapefruit White" ,"Guava" ,"Huckleberry" ,"Kaki" ,"Kiwi" ,"Kumquats" ,"Lemon" ,"Lemon Meyer" ,"Limes" ,"Litchi" ,"Mandarine" ,"Mango" ,"Maracuja" ,"Nectarine" ,"Orange" ,"Papaya" ,"Passion Fruit" ,"Peach" ,"Peach Flat" ,"Pear" ,"Pear Abate" ,"Pear Monster" ,"Pear Williams" ,"Pepino" ,"Pineapple" ,"Pitahaya Red" ,"Plum" ,"Pomegranate" ,"Quince" ,"Raspberry" ,"Salak" ,"Strawberry" ,"Tamarillo" , "Tangelo"]


test_image = image.load_img('banana.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
Training.class_indices
img = 'banana.jpg'
Image.open(img)


test_image = image.load_img('hb.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
Training.class_indices
img = 'hb.jpg'
Image.open(img)

test_image = image.load_img('litchi.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
Training.class_indices
img = 'litchi.jpg'
Image.open(img)

test_image = image.load_img('coco.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
Training.class_indices
img = 'coco.jpg'
Image.open(img)

test_image = image.load_img('gapp2.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
Training.class_indices
img = 'gapp2.jpg'
Image.open(img)

test_image = image.load_img('cf.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
Training.class_indices
img = 'cf.jpg'
Image.open(img)

test_image = image.load_img('lmn.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
Training.class_indices
img = 'lmn.jpg'
Image.open(img)


test_image = image.load_img('banana.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
Training.class_indices
img = 'banana.jpg'
Image.open(img)


for i in range(no_of_classes):
    if (result[0][i] == 1.0):
        img_path = 'Fruit-Images-Dataset/Training/'+name_of_classes[i]+'/2_100.jpg'
        print ('Predicted:',name_of_classes[i])
Image.open(img_path)