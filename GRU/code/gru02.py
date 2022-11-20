import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import TensorBoard

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('gruData') 
# Actions that we try to detect
actions = np.array(['squat-down','squat-up','pushup-down','pushup-up','lunge-down','lunge-up','stand'])


# Videos are going to be 30 frames in length
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}
path = '/Users/chaesiheon/Library/CloudStorage/OneDrive-성균관대학교/video_dataset' 
# sequences : x,y data | labels : what is it 
sequences, labels = [], []
for action in actions:
    path2=path+'/'+action
    file_list = os.listdir(path2) 
    no_sequences=len(file_list)
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence).zfill(4), "{}.npy".format(str(frame_num).zfill(3))))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
print(len(X))
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential() # 30 frame, 1662 length .npy file
model.add(GRU(64, return_sequences=True, activation='relu', input_shape=(30,48)))
# model.add(GRU(128, return_sequences=True, activation='relu'))
# model.add(GRU(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# train
model.fit(X_train, y_train, epochs=20, callbacks=[tb_callback])

model.summary()

model.save('gruaction.h5')