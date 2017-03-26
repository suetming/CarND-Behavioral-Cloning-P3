import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def random_drop(data):
    index = data[abs(data['steer'])<.05].index.tolist()
    rows = [i for i in index if np.random.randint(10) < 8]
    data = data.drop(data.index[rows])
    print("Dropped %s rows with low steering"%(len(rows)))
    return data

def preprocess(img_path):
    img = plt.imread(img_path)
    return img[60:135, : ]

def random_process(data, value, data_path):
    random = np.random.randint(4)
    if (random == 0):
        img_path = data['left'][value].strip()
        shift_ang = .25
    elif (random == 1 or random == 3):
        img_path = data['center'][value].strip()
        shift_ang = 0.
    elif (random == 2):
        img_path = data['right'][value].strip()
        shift_ang = -.25

    img = preprocess(os.path.join(data_path, img_path))
    angle = float(data['steer'][value]) + shift_ang
    return img, angle


def trans_image(image, steer):
    trans_range = 100
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 0
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (320,75))
    return image_tr, steer_ang


def generator0(data, batch_size, data_path):
    while 1:
        batch = data.sample(n=batch_size)
        features = np.empty([batch_size, 75, 320, 3])
        labels = np.empty([batch_size, 1])
        for i, value in enumerate(batch.index.values):
            # Randomly select right, center or left image
            img, steer_ang = random_process(data, value, data_path)
            img = img.reshape(img.shape[0], img.shape[1], 3)
            # Random Translation Jitter
            img, steer_ang = trans_image(img, steer_ang)
            # Randomly Flip Images
            random = np.random.randint(1)
            if (random == 0):
                img, steer_ang = np.fliplr(img), -steer_ang
            features[i] = img
            labels[i] = steer_ang
            yield np.array(features), np.array(labels)

def generator1(data, data_path):
    """
    Validation Generator
    """
    while 1:
        for i in range(len(data)):
            img_path = data['center'][i].strip()
            img = preprocess(os.path.join(data_path, img_path))
            img = img.reshape(1, img.shape[0], img.shape[1], 3)
            steer_ang = data['steer'][i]
            steer_ang = np.array([[steer_ang]])
            yield img, steer_ang
