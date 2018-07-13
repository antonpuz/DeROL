import numpy as np
import os
import random
import math

from scipy.ndimage import rotate,shift
from scipy.misc import imread,imresize


def get_shuffled_images(paths, labels, nb_samples=None, shuffle=True, last_class_offset=0, late_instances=1):
    if nb_samples is not None:
        before_third_samples = int(math.ceil(last_class_offset/len(labels)))
        all_but_last_samples = int(math.ceil(before_third_samples * float(len(labels)) / float(len(labels) - late_instances)))
        sampler_before = lambda x: random.sample(x, all_but_last_samples)
        sampler_after = lambda x: random.sample(x, nb_samples-before_third_samples)
    else:
        sampler_before = lambda x:x
        sampler_after  = lambda x:x

    # print "Sampling from: " + str(paths)
    images_before = [(i, os.path.join(path, image)) for i,path in zip(labels[late_instances:],paths[late_instances:]) for image in sampler_before(os.listdir(path)) ]
    images_after  = [(i, os.path.join(path, image)) for i,path in zip(labels,paths) for image in sampler_after(os.listdir(path)) ]

    if(shuffle):
        random.shuffle(images_before)
        random.shuffle(images_after)

    images = images_before + images_after
    images = images[0: len(labels) * nb_samples]
    return images

def time_offset_label(labels_and_images):
    labels, images = zip(*labels_and_images)
    time_offset_labels = (None,) + labels[:-1]
    return zip(images, time_offset_labels)

def load_transform(image_path, angle=0., s=(0,0), size=(20,20)):
    #Load the image
    original = imread(image_path, flatten=True)
    #Rotate the image
    rotated = np.maximum(np.minimum(rotate(original, angle=angle, cval=1.), 1.), 0.)
    #Shift the image
    shifted = shift(rotated, shift=s)
    #Resize the image
    resized = np.asarray(imresize(rotated, size=size), dtype=np.float32) / 255 #Note here we coded manually as np.float32, it should be tf.float32
    #Invert the image
    inverted = 1. - resized
    max_value = np.max(inverted)
    if max_value > 0:
        inverted /= max_value
    return inverted