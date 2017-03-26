import cv2
import numpy as np
from scipy.misc import imread, imresize
from sklearn.utils import shuffle

DATA_PATH = "../CarND-Behavioral-Cloning-P3-Other/data1"

def generator(samples, batch_size=32):
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images, angles = read_imgs(batch_samples)
            # yield shuffle(images, angles)
            batch_imgs, batch_angles = augment(preprocess(images), angles)
            yield shuffle(batch_imgs, batch_angles)

def read_imgs(samples):
    size = len(samples)
    images = np.empty([size * 3, 160, 320, 3])
    angles = np.empty([size * 3])

    for i, sample in enumerate(samples):
        images[3 * (i + 1) - 3] = imread(DATA_PATH + '/IMG/'+sample[0].split('/')[-1])
        images[3 * (i + 1) - 2] = imread(DATA_PATH + '/IMG/'+sample[1].split('/')[-1])
        images[3 * (i + 1) - 1] = imread(DATA_PATH + '/IMG/'+sample[2].split('/')[-1])

        # create adjusted steering measurements for the side camera images
        correction = 0.08 # this is a parameter to tune

        angles[3 * (i + 1) - 3] = float(sample[3])
        angles[3 * (i + 1) - 2] = float(sample[3]) + correction
        angles[3 * (i + 1) - 1] = float(sample[3]) - correction

    return images, angles

def preprocess(imgs):
    imgs = crop(imgs, cropping=((64, 24), (0, 0)))
    # imgs = resize(imgs, shape=(16, 32, 3))
    # imgs = rgb2gray(imgs)
    # imgs = normalize(imgs)

    return imgs

def normalize(imgs):
    """
    Normalize images between [-1, 1].
    """
    return imgs / (255.0 / 2) - 1

def rgb2gray(imgs):
    """
    Convert images to grayscale.
    """
    return np.mean(imgs, axis=3, keepdims=True)


def crop(imgs, cropping=((0,0),(0,0)), shape=(160, 320, 3)):
    """
    Crop useless information
    """
    height, width, channels = shape
    imgs_crop = np.empty([len(imgs), height - (cropping[0][0] + cropping[0][1]), width - (cropping[1][0] + cropping[1][1]), channels])
    for i, img in enumerate(imgs):
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        imgs_crop[i] = img[cropping[0][0]: height - cropping[0][1], cropping[1][0] : width - cropping[1][1]]
    return imgs_crop

def resize(imgs, shape=(6, 32, 3)):
    """
    Resize images to shape.
    """
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        imgs_resized[i] = imresize(img, shape)

    return imgs_resized

def random_flip(imgs, angles):
    """
    Augment the data by randomly flipping some angles / images horizontally.
    """
    new_imgs = np.empty_like(imgs)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(imgs, angles)):
        if np.random.choice(2):
            new_imgs[i] = np.fliplr(img)
            new_angles[i] = angle * -1
        else:
            new_imgs[i] = img
            new_angles[i] = angle

    return new_imgs, new_angles

def trans(imgs, angles):
    new_imgs = np.empty_like(imgs)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(imgs, angles)):
        trans_range = 100
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        new_angles[i] = angle + tr_x / trans_range * 2 * .2

        tr_y = 0
        M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        new_imgs[i] = cv2.warpAffine(img, M, (320,72))
    return new_imgs, new_angles

def augment(imgs, angles):
    imgs, angles = trans(imgs, angles)
    imgs, angles = random_flip(imgs, angles)

    return imgs, angles

def random_drop(samples):
    data = []
    for i, sample in enumerate(samples):
        if float(sample[3]) < .05 and np.random.choice(10) >= 8:
            data.append(sample)
    return data

if __name__ == '__main__':
    shape=(160, 320, 3)
    height, width, channels = shape
    name = 'data/IMG/'+ 'IMG/center_2016_12_01_13_33_02_662.jpg'.split('/')[-1]
    center_image = cv2.imread(name)

    cropping=((64,36),(0,0))
    crop_img = center_image[cropping[0][0]: height - cropping[0][1], cropping[1][0] : width - cropping[1][1]] # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
