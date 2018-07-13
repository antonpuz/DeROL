import numpy as np
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import paired_manhattan_distances
from sklearn.metrics.pairwise import paired_euclidean_distances



class UNSWClassifier(object):
    def __init__(self, number_of_classes, normal_memory_size=1000, mal_memory_size=50, max_distance=2.0, similarity='l2'):
        super(UNSWClassifier, self).__init__()
        self.number_of_classes = number_of_classes
        self.normal_memory_size = normal_memory_size
        self.mal_memory_size = mal_memory_size
        self.normal_labels = []
        self.mal_labels = []
        self.normal_data_storage = None
        self.mal_data_storage = None
        self.gnb = GaussianNB()
        self.max_distance = max_distance
        self.similarity = similarity

    def classify(self, data_v):
        if((len(self.normal_labels) + len(self.mal_labels)) < 2):
            print("Calling classification without enough image samples - returning zero vector")
            return np.zeros(self.number_of_classes)

        if(self.normal_data_storage == None):
            all_data = self.mal_data_storage
        elif(self.mal_data_storage == None):
            all_data = self.normal_data_storage
        else:
            all_data = self.normal_data_storage + self.mal_data_storage
        all_labels = self.normal_labels + self.mal_labels
        data_length = len(all_labels)

        min_distance = np.ones(self.number_of_classes) * self.max_distance

        if(self.similarity == 'cos'):
            distances = paired_cosine_distances(all_data, np.tile(data_v, (data_length, 1)))
        elif(self.similarity == 'l1'):
            distances = paired_manhattan_distances(all_data, np.tile(data_v, (data_length, 1)))
        else:
            distances = paired_euclidean_distances(all_data, np.tile(data_v, (data_length, 1)))


        for index_i in range(data_length):
            if (abs(distances[index_i]) < min_distance[all_labels[index_i]]):
                min_distance[all_labels[index_i]] = abs(distances[index_i])

        min_distance = self.max_distance - min_distance
        return min_distance

    def add_image_sample(self, data_v, label):
        if(label >= self.number_of_classes):
            print("image label must not be greater than number of classes - 1")
            return

        if(label == 0): #Normal
            if (self.normal_data_storage == None):
                self.normal_data_storage = [data_v]
                self.normal_labels = [label]
            else:
                self.normal_data_storage.append(data_v)
                self.normal_labels.append(label)

            if (len(self.normal_labels) > self.normal_memory_size):
                self.normal_data_storage = self.normal_data_storage[1:]
                self.normal_labels = self.normal_labels[1:]
        else:
            if (self.mal_data_storage == None):
                self.mal_data_storage = [data_v]
                self.mal_labels = [label]
            else:
                self.mal_data_storage.append(data_v)
                self.mal_labels.append(label)

            if (len(self.mal_labels) > self.mal_memory_size):
                self.mal_data_storage = self.mal_data_storage[1:]
                self.mal_labels = self.mal_labels[1:]

        if(self.normal_data_storage == None):
            all_data = self.mal_data_storage
        elif(self.mal_data_storage == None):
            all_data = self.normal_data_storage
        else:
            all_data = self.normal_data_storage + self.mal_data_storage
        all_labels = self.normal_labels + self.mal_labels

        self.gnb = GaussianNB()
        self.gnb = self.gnb.fit(np.array(all_data), np.array(all_labels))

    def distance_from_sample(self, all_data, single_samples):
        if(all_data == None or len(all_data) == 0):
            return None

        data_length = len(all_data)

        if(self.similarity == 'cos'):
            distances = paired_cosine_distances(all_data, np.tile(single_samples, (data_length, 1)))
        elif(self.similarity == 'l1'):
            distances = paired_manhattan_distances(all_data, np.tile(single_samples, (data_length, 1)))
        else:
            distances = paired_euclidean_distances(all_data, np.tile(single_samples, (data_length, 1)))

        min_distance = 5.0
        for index_i in range(data_length):
            if (abs(distances[index_i]) < min_distance):
                min_distance = abs(distances[index_i])

        return min_distance


    def get_database_size(self):
        return len(self.normal_labels + self.mal_labels)





