
# Import necessary libraries
from __future__ import print_function
import cv2
import numpy as np
import warnings
from matplotlib.pyplot import imread, imshow
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Define the VGG16 architecture with customizable input tensor and classes
def VGGupdated(input_tensor=None, classes=4):
    img_rows, img_cols = 224, 224
    img_channels = 3

    img_dim = (img_rows, img_cols, img_channels)

    img_input = Input(shape=img_dim)



    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    #fully connected layers
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create the model
    model = Model(inputs=img_input, outputs=x, name='VGGdemo')
    return model

# Create and compile the VGG model
model = VGGupdated(classes=4)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#consider room = class1 type img

dataset_path = os.listdir('./img/class1/')

room_types = os.listdir('./img/class1/')

print(room_types)
#print (room_types)  #what kinds of rooms are in this dataset

print("Types of rooms found: ", len(dataset_path))
rooms = []

for item in room_types:

 # Get all the file names
    all_rooms = os.listdir('./img/class1/')

 # Add them to the list
    for room in all_rooms:
        rooms.append((item, str('./img/class1' + '/' +item) + '/' + room))
print(rooms)



# Build a dataframe
rooms_df = pd.DataFrame(data=rooms, columns=['room type', 'image'])
room_types = rooms_df['room type'].unique()

print(rooms_df.head())
#print(rooms_df.tail())

# Let's check how many samples for each category are present
print("Total number of rooms in the dataset: ", len(rooms_df))

room_count = rooms_df['room type'].value_counts()

print("rooms in each category: ")
print(room_count)

img_size = 224


# Initialize the list to store image paths
images = []

images = []
labels = []
path = './img/class1/'
for i in room_types:
    data_path = path
    filenames = [i for i in os.listdir(data_path)]

    for f in filenames:
        img = cv2.imread(data_path+f)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(i)


# print(images)
# print(labels)

images = np.array(images)

images = images.astype('float32') / 255.0
print(images.shape)

y=rooms_df['room type'].values
#print(y[:5])

y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(rooms_df['room type'])
print(y)

onehotencoder = OneHotEncoder()
Y = onehotencoder.fit_transform(y.reshape(-1, 1)).toarray()
print(Y.shape)


images, Y = shuffle(images, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)

#inpect the shape of the training and testing.
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)

model.fit(train_x, train_y, epochs = 10, batch_size = 32)

preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
#print ("Test Accuracy = " + str(preds[1]))

img_path = './img/class1/mildDem0.jpg'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)


x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(model.predict(x))