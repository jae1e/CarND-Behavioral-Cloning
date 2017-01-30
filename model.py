import pickle
import math
import os
import h5py
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam

# load processed data
print("++ loading processed data")
pickle_filename = 'train.p'
with open(pickle_filename, 'rb') as p:
    pickle_data = pickle.load(p)
    X_train = pickle_data['features']
    y_train = pickle_data['labels']
pickle_filename = 'test.p'
with open(pickle_filename, 'rb') as p:
    pickle_data = pickle.load(p)
    X_test = pickle_data['features']
    y_test = pickle_data['labels']
pickle_filename = 'validation.p'
with open(pickle_filename, 'rb') as p:
    pickle_data = pickle.load(p)
    X_validation = pickle_data['features']
    y_validation = pickle_data['labels']
del pickle_data
print("-- loaded processed data")
print("-- train set size : ", len(X_train))
print("-- test set size : ", len(X_test))
print("-- validation set size : ", len(X_validation))
image_shape = X_train[0].shape
print("-- image shape : ", image_shape)
print()

#######################################################################

batch_size = 64 # The lower the better
n_classes = 1 # The output is a single digit: a steering angle
n_epoch = 10 # The higher the better

json_filename = "model.json"
h5_filename = "model.h5"

# import or create model
print("++ import or create model")

create_model = False

try:
	with open(json_filename, 'r') as jfile:
		print("++ model file already exist")
		print("++ do you want to import model? y or n")
		user_input1 = input()
		if user_input1 == "y":
			# import model
		    model = model_from_json(json.load(jfile))
		    # use adam and mean squared error for training
		    model.compile("adam", "mse")
		    print("-- model imported")

		    # import weights
		    print("++ weights file already exist")
		    print("++ do you want to import wieghts? y or n")
		    user_input2 = input()
		    if user_input2 == "y":
		    	model.load_weights(h5_filename)
		    	print("-- weights imported")
		else:
			create_model = True
except:
	create_model = True

# create model
if create_model == True:
	n_filters1 = 16
	n_filters2 = 24
	n_filters3 = 32
	n_filters4 = 64
	pool_size = (2, 2)
	kernel_size = (3, 3)
	p_drop1 = 0.25
	p_drop2 = 0.25
	n_fc1 = 128
	n_fc2 = 64
	n_fc3 = 32

	model = Sequential([
	# convolution 1 - input: 1, output: 16
	Conv2D(n_filters1, kernel_size[0], kernel_size[1], input_shape=image_shape, border_mode='same', activation='relu'),
	# convolution 2 - input: 16, output: 8
	Conv2D(n_filters2, kernel_size[0], kernel_size[1], border_mode='same', activation='relu'),
	# convolution 3 - input: 8, output: 4
	Conv2D(n_filters3, kernel_size[0], kernel_size[1], border_mode='same', activation='relu'),
	# convolution 4 - input: 4, output: 2
	Conv2D(n_filters4, kernel_size[0], kernel_size[1], border_mode='same', activation='relu'),
	# pooling 1 - 2 x 2
	MaxPooling2D(pool_size=pool_size),
	# dropout 1 - 25%
	Dropout(p_drop1),
	# flatten
	Flatten(),
	# fully connected 1 - input ?, output 16
	Dense(n_fc1, activation='relu'),
	# fully connected 2 - input 16, output 16
	Dense(n_fc2, activation='relu'),
	# fully connected 3 - input 16, output 16
	Dense(n_fc3, activation='relu'),
	# dropout 2 - 50%
	Dropout(p_drop2),
	# fully connected 4 - input 16, output 1
	Dense(n_classes)
	])

	print("-- model constructed")

# show summary of the model
model.summary()

# compile model
model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

# train model
print("++ training model")
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=n_epoch,
                    verbose=1, validation_data=(X_validation, y_validation),
                    shuffle=True)
# print(y_train)

print("-- model trained")
print()
print("++ testing model")
evaluation = model.evaluate(X_test, y_test, verbose=0)
print("-- testing done")
print('-- score:', evaluation[0])
print('-- accuracy:', evaluation[1])
for i in range(0, len(X_test), int(len(X_test) / 10)):
	test_image = X_test[i]
	test_image.resize(1, image_shape[0], image_shape[1], image_shape[2])
	print(test_image.shape)
	steering_angle = float(model.predict(test_image, batch_size=1))
	print("-- example prediction: ", steering_angle)

# save model
if json_filename in os.listdir():
	print("++ model file already exists")
	print("++ do you want to overwite? y or n")
	user_input = input()

	if user_input == "y":
		# save model
		json_string = model.to_json()

		with open(json_filename, 'w') as outfile:
			json.dump(json_string, outfile)

		# save weights
		model.save_weights(h5_filename)
		print("-- model overwrited successfuly")
	else:
		print("-- model not saved")
else:
	# save model
	json_string = model.to_json()

	with open(json_filename, 'w') as outfile:
		json.dump(json_string, outfile)

	# save weights
	model.save_weights(h5_filename)
	print("-- model saved")
