from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.spatial.distance import cdist
import numpy as np
from multiprocessing import Pool as ThreadPool

from classifiers import Images


def _ModHausdorffDistance(itemA, itemB):
    D = cdist(itemA, itemB)
    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)
    mean_A = np.mean(mindist_A)
    mean_B = np.mean(mindist_B)
    return max(mean_A, mean_B)

def break_param_names_and_run_distance(param):
    return (param[0], _ModHausdorffDistance(param[1], param[2]))

class OMNIGLOTClassifier(object):
    def __init__(self, number_of_classes, memory_size=100, number_of_executors=100, image_size=(20,20)):
        super(OMNIGLOTClassifier, self).__init__()
        self.number_of_classes = number_of_classes
        self.image_size = image_size
        self.memory_size = memory_size
        self.image_names = np.array([])
        self.image_loads = {}
        self.image_labels = {}
        self.different_labels = {}
        self.representing_image_cache = {}
        self.image_points_cache = {}
        self.executionPool = ThreadPool(min(number_of_executors, memory_size))

    def LoadImgAsPoints(self, fn):
        if(fn in self.image_points_cache):
            return self.image_points_cache[fn]

        I = imread(fn, flatten=True)
        I = np.asarray(imresize(I, size=self.image_size), dtype=np.float32)
        I[I<255] = 0
        I = np.array(I, dtype=bool)
        I = np.logical_not(I)
        (row, col) = I.nonzero()
        D = np.array([row, col])
        D = np.transpose(D)
        D = D.astype(float)
        n = D.shape[0]
        mean = np.mean(D, axis=0)
        for i in range(n):
            D[i, :] = D[i, :] - mean

        self.image_points_cache[fn] = D
        return D

    def load_representing_image(self, fn):
        if(fn in self.representing_image_cache):
            return self.representing_image_cache[fn]

        I = Images.load_transform(fn, size=self.image_size)

        self.representing_image_cache[fn] = I

        return I

    def classify(self, image_file):
        if(len(self.image_names) == 0):
            print("Calling classification without enough image samples - returning zero vector")
            return np.zeros(self.number_of_classes)

        points_image = self.LoadImgAsPoints(image_file)
        min_distance = np.ones(self.number_of_classes) * 2.0
        unbraked_values = [(self.image_labels[image], self.LoadImgAsPoints(image), points_image) for image in self.image_names]
        distances = self.executionPool.map(break_param_names_and_run_distance, unbraked_values)

        for distance in distances:
            if (distance[1] < min_distance[distance[0]]):
                min_distance[distance[0]] = distance[1]

        min_distance = 2.0 - min_distance
        return min_distance

    def add_image_sample(self, image_file, label):
        if(label >= self.number_of_classes):
            print("image label must not be greater than number of classes - 1")
            return

        if(image_file in self.image_loads):
            old_label = self.image_labels[image_file]
            if (self.different_labels[old_label] == 1):
                del self.different_labels[old_label]
            else:
                self.different_labels[old_label] -= 1

        if(not image_file in self.image_loads):
            self.image_names = np.append(self.image_names, image_file)
        if (label in self.different_labels):
            self.different_labels[label] += 1
        else:
            self.different_labels[label] = 1
        self.image_loads[image_file] = self.LoadImgAsPoints(image_file)
        self.image_labels[image_file] = label

        to_be_deleted = None
        if(len(self.image_names) > self.memory_size):
            to_be_deleted = self.image_names[0]
            self.image_names = self.image_names[1:]

        if(to_be_deleted != None):
            del self.image_loads[to_be_deleted]
            removed_label = self.image_labels[to_be_deleted]
            if(not removed_label in self.different_labels):
                print("error found")
            if (self.different_labels[removed_label] == 1):
                del self.different_labels[removed_label]
            else:
                self.different_labels[removed_label] -= 1
            del self.image_labels[to_be_deleted]

        if(sum(self.different_labels.values()) > self.memory_size):
            print("classifier reached an unexpected")

        if (len(self.image_loads) != len(self.image_labels) or len(self.image_labels) != len(self.image_names)):
            print("classifier reached an unexpected")

    def database_size(self):
        return len(self.different_labels)






