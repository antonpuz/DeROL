import numpy as np
import os
import random

from classifiers.Images import load_transform, get_shuffled_images


class OMNIGLOTGenerator(object):
    """Docstring for OmniglotGenerator"""
    def __init__(self, data_folder, letter_swap=1, batch_size=1, classes=5, samples_per_class=10, max_rotation=0., max_shift=0., img_size=(20,20), number_of_classes=30, max_iter=None, only_labels_and_images=False):
        super(OMNIGLOTGenerator, self).__init__()
        self.data_folder = data_folder
        self.letter_swap = letter_swap
        self.batch_size = batch_size
        self.number_of_classes = number_of_classes
        self.classes = classes
        self.samples_per_class = samples_per_class
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        self.img_size = img_size
        self.max_iter = max_iter
        self.num_iter = 0
        self.only_labels_and_images = only_labels_and_images
        self.character_folders = [os.path.join(self.data_folder, family, character) \
                                  for family in os.listdir(self.data_folder) \
                                  if os.path.isdir(os.path.join(self.data_folder, family)) \
                                  for character in os.listdir(os.path.join(self.data_folder, family))]
        self.working_characters = random.sample(self.character_folders, self.classes)
        self.working_labels = np.random.choice(self.number_of_classes, self.classes, replace=False).tolist()
        self.cacheDict = {}
        self.newest_swapped_letter = self.working_labels[0]


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1), self.sample(self.classes)
        else:
            raise StopIteration

    def _get_class_for_folders(self, working_chars):
        char_labels = np.random.choice(range(self.number_of_classes), len(working_chars), replace=False)
        return char_labels.tolist()


    def get_image_by_name_and_chars(self, filename, angle, shift, img_size):
        e_key = (filename, angle, shift, img_size)
        if e_key in self.cacheDict:
            # print "cache hit! with " + str(e_key)
            # print "returning result of size " + str(self.cacheDict[e_key].shape)
            return self.cacheDict[e_key]
        else:
            # print "cache miss, adding " + str(e_key)
            obtained_image = load_transform(filename, angle=angle, s=shift, size=img_size).flatten()
            self.cacheDict[e_key] = obtained_image
            return obtained_image

    def sample(self, nb_samples):
        # print "character_folders"
        # print self.character_folders
        # print "nb_samples"
        # print nb_samples
        # sampled_character_folders = random.sample(self.character_folders, nb_samples)
        for i in range(self.letter_swap):
            index_to_swap = random.randint(0, nb_samples-1)
            self.newest_swapped_letter = self.working_labels[index_to_swap]
            self.working_characters[index_to_swap] = random.sample(self.character_folders, 1)[0]
            self.working_labels[index_to_swap] = np.random.randint(0,self.number_of_classes,1)[0]

        # print "Creating sample sets from " + str(self.working_characters)
        # random.shuffle(self.working_characters)

        example_inputs = np.zeros((self.batch_size, nb_samples * self.samples_per_class, np.prod(self.img_size)), dtype=np.float32)
        example_outputs = np.zeros((self.batch_size, nb_samples * self.samples_per_class), dtype=np.float32)     #notice hardcoded np.float32 here and above, change it to something else in tf
        folder_and_labels_only_list = []

        for i in range(self.batch_size):
            labels_and_images = get_shuffled_images(self.working_characters, self.working_labels, nb_samples=self.samples_per_class)
            if(self.only_labels_and_images):
                if (folder_and_labels_only_list == []):
                    folder_and_labels_only_list = [labels_and_images]
                else:
                    folder_and_labels_only_list = np.vstack((folder_and_labels_only_list, [labels_and_images]))
                continue

            sequence_length = len(labels_and_images)
            labels, image_files = zip(*labels_and_images)

            angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=sequence_length)
            shifts = np.random.uniform(-self.max_shift, self.max_shift, size=sequence_length)

            example_inputs[i] = np.asarray([self.get_image_by_name_and_chars(filename, angle=angle, shift=shift, img_size=self.img_size).flatten() \
                                            for (filename, angle, shift) in zip(image_files, angles, shifts)], dtype=np.float32)
            example_outputs[i] = np.asarray(labels, dtype=np.int32)

        # print "Created input sets of size " + str(example_inputs.shape)
        if (self.only_labels_and_images):
            return folder_and_labels_only_list
        else:
            return example_inputs, example_outputs

    def get_last_swapped_letter(self):
        return self.newest_swapped_letter


