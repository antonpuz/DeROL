from multiprocessing import Pool as ThreadPool
import numpy as np

from classifiers.OMNIGLOTClassifier import break_param_names_and_run_distance


class AnalystManagement:
    def __init__(self, number_of_analysts, delay_creator, image_classifier, number_of_threads=20, maximal_delay_size=10000000):
        self.delay_creator = delay_creator
        self.maximal_size = maximal_delay_size
        if maximal_delay_size<=0:
            raise ValueError('maximal value for iterations should be positive, obtained {}'.format(maximal_delay_size))
        self.current_delay_position = 0
        self.delayed_classifications = {}
        self.number_of_analysts = number_of_analysts
        self.busy_analysts = 0
        self.job_queue = []
        # self.job_queue = Queue.Queue()
        self.image_classifier = image_classifier
        self.executionPool = ThreadPool(number_of_threads)

    def __add_item(self, item_to_add, delay=0):
        # Add element to be written
        position_after_delay = (self.current_delay_position + delay) % self.maximal_size
        to_be_written_after_delay = self.delayed_classifications.get(position_after_delay, [])
        to_be_written_after_delay.append(item_to_add)
        self.delayed_classifications[position_after_delay] = to_be_written_after_delay

    def __start_processing_of_job(self, buffer): #private
        wait_for = self.delay_creator.get_rand_delay()
        self.busy_analysts += 1
        self.__add_item(buffer, delay=wait_for)

    def analysts_load(self):
        return float(self.busy_analysts + len(self.job_queue))/self.number_of_analysts

    def add_classification_job(self, buffer):
        if self.busy_analysts<self.number_of_analysts:
            self.__start_processing_of_job(buffer)
        else:
            self.job_queue.append(buffer)


    def free_job(self):
        self.busy_analysts -= 1
        # print("In NoUpdateRegularAnalysts: free job, new analyst load: " + str(self.busy_analysts))
        if not len(self.job_queue) == 0:
            sample_to_return = self.job_queue[0]
            self.job_queue = self.job_queue[1:]
            self.__start_processing_of_job(sample_to_return)


    def advance_time(self):
        self.current_delay_position = (self.current_delay_position + 1) % self.maximal_size
        # Add elements delayed for now
        if self.current_delay_position in self.delayed_classifications:
            # print("In DelayedInsertions(" + str(self.current_position) + "): got some classified items")
            elements_to_add = self.delayed_classifications.get(self.current_delay_position)
            for element in elements_to_add:
                self.free_job()
                if self.image_classifier != None:
                    self.image_classifier.add_image_sample(element[0], element[1])
            del self.delayed_classifications[self.current_delay_position]

    def distance_to_samples_in_work(self, image_file):
        points_image = self.image_classifier.LoadImgAsPoints(image_file)
        unbraked_values = [(y, self.image_classifier.LoadImgAsPoints(sample_file), points_image) for (sample_file, y) in
                           self.job_queue]
        for key in self.delayed_classifications.keys():
            processed_elements = self.delayed_classifications.get(key)
            for element in processed_elements:
                unbraked_values.append((element[1], self.image_classifier.LoadImgAsPoints(element[0]), points_image))

        distances = self.executionPool.map(break_param_names_and_run_distance, unbraked_values)
        clean_distances = np.array([res[1] for res in distances])
        if(len(clean_distances) > 0):
            res = 1.0 - min(clean_distances)
            return res
        else:
            return 0.0

