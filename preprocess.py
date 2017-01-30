import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cross_validation import train_test_split
import pickle
import util

features = []
labels = []
features_test = []
labels_test = []

# folders = ["./sample-data/data", 
#             "./track1-normal1", 
#             "./track1-normal4", 
#             "./track1-normal2", 
#             "./track1-normal3", 
#             "./track1-normal5",
#             "./track1-opencorner1",
#             "./track1-opencorner2"]
# folders = ["./sample-data/data", "./track1-normal1", "./track1-normal4", "./track1-opencorner1", "./track1-opencorner2"]
folders = ["./sample-data/data", 
            "./track1-normal1", 
            "./track1-normal4", 
            "./track1-normal2", 
            "./track1-normal5"]

# generate data
for folder_path in folders:
    folder_path += "/"
    data_file_path = folder_path + "driving_log.csv"

    # import data
    print("++ importing data from ", folder_path)
    data = []
    with open(data_file_path) as F:
        reader = csv.reader(F)
        for i in reader:
            data.append(i) 
    data = data[1:]

    print("-- data imported")
    print("-- data size: ", len(data))
    print()

    # read and preprocess image data
    # generate features and labels
    print("++ generating features and labels")
    for i in tqdm(range(int(len(data))), unit='items'):
        angle = float(data[i][3]) 
        for j in range(3):
            image_filename = data[i][j].strip()
            if image_filename == "":
                continue
            pos = image_filename.find("IMG")
            if pos < 0:
                continue
            image_filename = image_filename[pos:]
            image_path = folder_path + image_filename
            image = plt.imread(image_path)
            image = util.preprocess_image(image)
            features.append(image)
            labels.append(angle)
            # add flipped image to balance left and right turn
            flipped = util.flip_image(image)
            features.append(flipped)
            labels.append(-angle)

features = np.array(features)
labels = np.array(labels)

print("-- features and labels generated")
print("-- features shape : ", features.shape)
print("-- labels shape : ", labels.shape)
print()

# generate train, test, validation set
print("++ generating test, train, validation set")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.10, random_state=27457)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.10, random_state=27457)

print("-- test, train, validation set generated")
print("-- train set size : ", len(X_train))
print("-- test set size : ", len(X_test))
print("-- validation set size : ", len(X_validation))
print()

# save processed data
print("++ saving processed data")
pickle_filename = 'train.p'
with open(pickle_filename, 'wb') as p:
    pickle.dump(
        {
            'features': X_train,
            'labels': y_train,
        },
        p, pickle.HIGHEST_PROTOCOL)
print("-- train data saved in ", pickle_filename)
pickle_filename = 'test.p'
with open(pickle_filename, 'wb') as p:
    pickle.dump(
        {
            'features': X_test,
            'labels': y_test,
        },
        p, pickle.HIGHEST_PROTOCOL)
print("-- test data saved in ", pickle_filename)
pickle_filename = 'validation.p'
with open(pickle_filename, 'wb') as p:
    pickle.dump(
        {
            'features': X_validation,
            'labels': y_validation,
        },
        p, pickle.HIGHEST_PROTOCOL)
print("-- validation data saved in ", pickle_filename)
print()
