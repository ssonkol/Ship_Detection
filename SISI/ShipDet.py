import json, sys, random
import numpy as np
import tensorflow
from keras.models import sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utilis import np_utils
from keras.optimizers import SGD
import keras.callbacks

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


f = open("D:\\Data\\Ships in Satellite Imagery\\shipsnet.json")#opens files in directory
dataset = json.load(f)#loads the dataset and assigns it to the variable
f.close()#closes file

inputData = np.array(dataset['data']).astype('uint8')
outputData = np.array(dataset['labels']).astype('uint8')

inputData.shape

n_spectrum = 3
weight = 80
height = 80
X = inputData.reshape([-1, n_spectrum, weight, height])
X[0].shape
pic = X[0]
rad_spectrum = pic[0]
green_spectrum = pic[1]
blue_spectrum = pic[2]

plt.figure(2, figsize = (5*3, 5*1))
plt.set_cmap('jet')

plt.subplot(1, 3, 2)
plt.imshow(green_spectrum)

plt.subplot(1, 3, 3)
plt.imshow(blue_spectrum)
plt.show()

outputData.shape
outputData

np.bincount(outputData)


#Preparing Data
y = np_utils.to_categorical(outputData, 2) #ouput encoding
#shuffle all indexes
indexes = np.arange(2800)
np.random.shuffle(indexes)
Xtrain = X[indexes].transpose([0,2,3,1])
Ytrain = y[indexes]
#Normailsation of vectors
Xtrain =Xtrain/255

#Training Network

np.random.seed(42)
#network design
model = sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
model.add(Dropout(0.25))

model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# optimization setup
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy'])

# training
model.fit(
    Xtrain, 
    Ytrain,
    batch_size=32,
    epochs=18,
    validation_split=0.2,
    shuffle=True,
    verbose=2)


#Downloading image
image = Image.open('D:\\Data\\sfbay.png\\sfbay_1.png')
pix = image.load()

n_spectrum = 3
width = image.size[0]
height = image.size[1]

#create Vector
pictureVector = []
for chanel in range(n_spectrum):
    for y in range(height):
        for x in range(width):
            pictureVector.append(pix[x, y][chanel])

pitcureVector =np.array(pictureVector).astype('uint8')
picture_tensor = pitcureVector.reshape([n_spectrum, height, width]).transpose(1, 2, 0)
plt.figure(1, figsize = (15, 30))

plt.subplot(2, 1, 1)
plt.imshow(picture_tensor)
plt.show()

picture_tensor = picture_tensor.transpose(2, 0, 1)

#Search on the image
def cutting(x, y):
    area_study = np.arange(3*80*80).reshape(3, 80, 80)
    for i in range(80):
        for j in range(80):
            area_study[0][i][j] = picture_tensor[0][y+i][x+j]
            area_study[1][i][j] = picture_tensor[1][y+i][x+j]
            area_study[2][i][j] = picture_tensor[2][y+i][x+j]
    area_study = area_study.reshape([-1, 3, 80, 80])
    area_study = area_study.transpose([0,2,3,1])
    area_study = area_study / 255
    sys.stdout.write('\rX:{0} Y:{1}  '.format(x, y))
    return area_study

def not_near(x, y, s, coordinates):
    result = True
    for e in coordinates:
        if x+s > e[0][0] and x-s < e[0][0] and y+s > e[0][1] and y-s < e[0][1]:
            result = False
    return result

def show_ship(x, y, acc, thickness=5):   
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x-th] = -1

    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x+th+80] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y-th][x+i] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+th+80][x+i] = -1

step = 10; coordinates = []
for y in range(int((height-(80-step))/step)):
    for x in range(int((width-(80-step))/step) ):
        area = cutting(x*step, y*step)
        result = model.predict(area)
        if result[0][1] > 0.90 and not_near(x*step,y*step, 88, coordinates):
            coordinates.append([[x*step, y*step], result])
            print(result)
            plt.imshow(area[0])
            plt.show()

for e in coordinates:
    show_ship(e[0][0], e[0][1], e[1][0][1])

picture_tensor = picture_tensor.transpose(1,2,0)
picture_tensor.shape

plt.figure(1, figsize = (15, 30))

plt.subplot(3,1,1)
plt.imshow(picture_tensor)
plt.show()