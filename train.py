import cv2
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 64
LR = 1e-4

MODEL_NAME = 'dogsvscats.model'


def label_image(image):
	word_label = image.split('.')[0]
	if word_label == 'cat':
		return [1, 0]
	elif word_label == 'dog':
		return [0,1]

def create_train_data():
	train_data = []
	for image in os.listdir(TRAIN_DIR):
		label = label_image(image)
		path = os.path.join(TRAIN_DIR, image)
		image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
		train_data.append([np.array(image), np.array(label)])
	shuffle(train_data)
	np.save('train_data.npy', train_data)
	return train_data

train_data = create_train_data()


convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
print(convnet.shape)

convnet = conv_2d(convnet, 32, 3, activation='relu')
print(convnet.shape)
convnet = max_pool_2d(convnet, 2)
print(convnet.shape)

convnet = conv_2d(convnet, 64, 3, activation='relu')
print(convnet.shape)
convnet = max_pool_2d(convnet, 2)
print(convnet.shape)

convnet = conv_2d(convnet, 128, 3, activation='relu')
print(convnet.shape)
convnet = max_pool_2d(convnet, 2)
print(convnet.shape)

convnet = conv_2d(convnet, 256, 3, activation='relu')
print(convnet.shape)
convnet = max_pool_2d(convnet, 2)
print(convnet.shape)

convnet = conv_2d(convnet, 128, 3, activation='relu')
print(convnet.shape)
convnet = max_pool_2d(convnet, 2)
print(convnet.shape)

convnet = conv_2d(convnet, 64, 3, activation='relu')
print(convnet.shape)
convnet = max_pool_2d(convnet, 2)
print(convnet.shape)

convnet = conv_2d(convnet, 32, 3, activation='relu')
print(convnet.shape)
convnet = max_pool_2d(convnet, 2)
print(convnet.shape)

convnet = fully_connected(convnet, 512, activation='relu')
print(convnet.shape)
convnet = dropout(convnet, 0.75)

convnet = fully_connected(convnet, 2, activation='softmax')
print(convnet.shape)
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='logs/incr_epochs/5epochs', tensorboard_verbose=3)

# Use 20% of the data for validation
train_len = int(len(os.listdir(TRAIN_DIR)) * 0.8)
print(train_len)
train = train_data[:train_len]
test = train_data[train_len:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3,
    validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)


#testing data

def process_test_data():
	test_data = []
	for image in os.listdir(TEST_DIR):
		path = os.path.join(TEST_DIR, image)
		label = label_image(image)
		image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
		test_data.append([np.array(image), label])

	shuffle(test_data)
	np.save('test_data.npy', test_data)
	return test_data

test_data = process_test_data()

fig = plt.figure()

for num, data in enumerate(test_data):
	img_data = data[0]
	true_label = data[1]

	y = fig.add_subplot(5,4,num+1)
	orig = img_data
	data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
	model_out = model.predict([data])[0]

	if (np.argmax(model_out) == 1):
		pred_label = 'dog'
	else:
		pred_label = 'cat'


	y.imshow(orig, cmap='gray')
	plt.title(pred_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()


