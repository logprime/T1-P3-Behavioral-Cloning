import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
  next(csvfile, None)
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):
        # Load images from center, left and right cameras
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = "./data/IMG/" + filename
        image = cv2.imread(local_path)
        images.append(image)

    # Introduce steering correction Thanks to Vivek Yadav's suggestion in his post on medium
    correction = 0.25
    measurement = float(line[3])
    # Steering adjustment for center images
    measurements.append(measurement)
    # Add correction for steering for left images
    measurements.append(measurement+correction)
    # Minus correction for steering for right images
    measurements.append(measurement-correction)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=7)

model.save('clone.h5')
