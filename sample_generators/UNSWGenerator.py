import numpy as np
import os
import random
import csv

class UNSWGenerator(object):
    # STD
    # [3.11718494e+00, 5.05267629e+04, 1.59565885e+05, 6.01614215e+01, 3.68997951e+01, 2.04402226e+01, 5.60362463e+01, 9.02498798e+07, 4.38022457e+06, 7.34661981e+01, 1.20085728e+02, 1.21618092e+02, 1.21674944e+02, 1.43249037e+09, 1.43025723e+09, 1.53408714e+02, 3.38400001e+02, 3.48792432e-01, 5.02959031e+04, 1.62749588e+04, 2.81668871e+03, 3.03954671e+03, 1.58047419e+03, 3.18336467e-02, 1.71418292e-02, 1.64202905e-02, 4.47561323e-02, 5.28876325e-01, 3.88885807e-01, 1.35820667e-01, 1.37266282e-01, 7.48326410e+00, 7.43839560e+00, 5.30909327e+00, 5.48918817e+00, 5.40727891e+00, 3.86212078e+00, 7.39321762e+00]
    # MEAN
    # [5.83990359e-01, 4.43920514e+03, 3.82112115e+04, 4.98032817e+01, 3.16409877e+01, 5.55065064e+00, 1.73402538e+01, 2.01082635e+07, 2.74214517e+06, 3.56218705e+01, 4.57140918e+01, 1.65776746e+02, 1.65596682e+02, 1.39622413e+09, 1.39395230e+09, 1.25076663e+02, 3.05978873e+02, 9.58984344e-02, 4.96244366e+03, 1.58372001e+03, 7.78032492e+02, 2.25335137e+02, 9.18970630e+01, 4.39537934e-03, 2.37087431e-03, 2.02450502e-03, 2.00713999e-03, 1.54769779e-01, 1.05521278e-01, 1.82014026e-02, 1.83442595e-02, 6.72776896e+00, 6.54925922e+00, 4.51331784e+00, 5.03591995e+00, 2.60927913e+00, 1.92820439e+00, 4.06247991e+00]

    def __init__(self, data_folder, letter_swap=1, batch_size=1, classes=5, samples_per_class=10, max_iter=None, train=True):
        super(UNSWGenerator, self).__init__()
        self.data_folder = data_folder
        self.trainLabelDict = {"Normal":0, "Exploits":1, "Reconnaissance":1, "DoS":1, \
                               "Generic": 1, "Shellcode":1, "Analysis":1}
        self.trainAttackDict = {"Exploits": 1, "Reconnaissance": 1, "DoS": 1, \
                               "Generic": 1, "Shellcode": 1, "Analysis": 1}

        self.testLabelDict = {"Normal_test":0, "Backdoors_test":1, "Fuzzers_test": 1, "Worms_test":1,
                              "Generic_test": 1, "Shellcode_test":1, "Analysis_test":1}
        self.testAttackDict = {"Backdoors_test": 1, "Fuzzers_test": 1, "Worms_test": 1,
                              "Generic_test": 1, "Shellcode_test": 1, "Analysis_test": 1}

        if(train):
            self.possible_labels = ["Normal"] + list(self.trainAttackDict)
            self.active_dir = self.trainLabelDict
        else:
            self.possible_labels = ["Normal_test"] + list(self.testAttackDict)
            self.active_dir = self.testLabelDict
        self.all_data_in_memory = {}

        for label in self.possible_labels:
            file = os.path.join(self.data_folder, label + "_UNSW.csv")
            print("reading from {}".format(file))
            with open(file, 'rb') as csvfile:
                spamreader = csv.reader(csvfile, quotechar='|')
                all_relevant_data = [row for row in spamreader]
                self.all_data_in_memory[label] = all_relevant_data


        self.letter_swap = letter_swap
        self.batch_size = batch_size
        self.classes = classes
        self.samples_per_class = samples_per_class
        self.max_iter = max_iter
        self.num_iter = 0
        self.features = 38
        self.working_labels = []
        for i in range(self.classes - 2):
            new_index = random.randint(0, len(self.possible_labels) - 1)
            self.working_labels.append(self.possible_labels[new_index])
        for i in range(2):
            if(train):
                self.working_labels.append("Normal")
            else:
                self.working_labels.append("Normal_test")
        self.cacheDict = {}
        self.newest_swapped_letter = self.working_labels[0]
        self.sample_mean = np.array([5.83990359e-01, 4.43920514e+03, 3.82112115e+04, 4.98032817e+01, 3.16409877e+01, 5.55065064e+00, 1.73402538e+01, 2.01082635e+07, 2.74214517e+06, 3.56218705e+01, 4.57140918e+01, 1.65776746e+02, 1.65596682e+02, 1.39622413e+09, 1.39395230e+09, 1.25076663e+02, 3.05978873e+02, 9.58984344e-02, 4.96244366e+03, 1.58372001e+03, 7.78032492e+02, 2.25335137e+02, 9.18970630e+01, 4.39537934e-03, 2.37087431e-03, 2.02450502e-03, 2.00713999e-03, 1.54769779e-01, 1.05521278e-01, 1.82014026e-02, 1.83442595e-02, 6.72776896e+00, 6.54925922e+00, 4.51331784e+00, 5.03591995e+00, 2.60927913e+00, 1.92820439e+00, 4.06247991e+00])
        self.sample_std = np.array([[3.11718494e+00, 5.05267629e+04, 1.59565885e+05, 6.01614215e+01, 3.68997951e+01, 2.04402226e+01, 5.60362463e+01, 9.02498798e+07, 4.38022457e+06, 7.34661981e+01, 1.20085728e+02, 1.21618092e+02, 1.21674944e+02, 1.43249037e+09, 1.43025723e+09, 1.53408714e+02, 3.38400001e+02, 3.48792432e-01, 5.02959031e+04, 1.62749588e+04, 2.81668871e+03, 3.03954671e+03, 1.58047419e+03, 3.18336467e-02, 1.71418292e-02, 1.64202905e-02, 4.47561323e-02, 5.28876325e-01, 3.88885807e-01, 1.35820667e-01, 1.37266282e-01, 7.48326410e+00, 7.43839560e+00, 5.30909327e+00, 5.48918817e+00, 5.40727891e+00, 3.86212078e+00, 7.39321762e+00]])


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

    def sample(self, nb_samples):
        for i in range(self.letter_swap):
            index_to_swap = random.randint(0, self.classes - 3)
            new_index = random.randint(0, len(self.possible_labels)-1)
            self.newest_swapped_letter = self.working_labels[0]
            self.working_labels[index_to_swap] = self.possible_labels[new_index]

        example_inputs = np.zeros((self.batch_size, nb_samples * self.samples_per_class, self.features), dtype=np.float32)
        example_outputs = np.zeros((self.batch_size, nb_samples * self.samples_per_class), dtype=np.float32)     #notice hardcoded np.float32 here and above, change it to something else in tf

        for i in range(self.batch_size):
            labels_and_samples = [(self.active_dir[active_label], x) for active_label in self.working_labels for x in random.sample(self.all_data_in_memory[active_label], self.samples_per_class)]
            random.shuffle(labels_and_samples)

            sequence_length = len(labels_and_samples)
            labels, samples = zip(*labels_and_samples)
            samples = np.asarray(samples, dtype=np.float32)
            #normalize
            samples = (samples - self.sample_mean) / self.sample_std

            example_inputs[i] = samples
            example_outputs[i] = np.asarray(labels, dtype=np.int32)

        return example_inputs, example_outputs

    def get_last_swapped_letter(self):
        return self.newest_swapped_letter


