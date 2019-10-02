import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pickle

FILE_PATH = "../dataset"

dataset = []

for PATH in os.listdir(FILE_PATH):
    directory_path = os.path.join(FILE_PATH, PATH)

    for FOLDER in os.listdir(directory_path):
        directory_folder = os.path.join(directory_path, FOLDER)

        for FILE_NAME in os.listdir(directory_folder):
            directory_filename = os.path.join(directory_folder, FILE_NAME)
            img = cv2.imread(directory_filename, cv2.IMREAD_COLOR)
            if FOLDER == "bird":
                dataset.append([img, 1])
            else:
                dataset.append([img, 0])

random.shuffle(dataset)

features = []
labels = []
for feature, label in dataset:
    features.append(feature)
    labels.append(label)

features = np.array(features)
labels = np.array(labels)

pickle_out = open("features.pickle", "wb")
pickle.dump(features, pickle_out)
pickle_out.close()

pickle_out = open("labels.pickle", "wb")
pickle.dump(labels, pickle_out)
pickle_out.close()