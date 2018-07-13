
class UNSWAnalystManagement:
    def __init__(self, number_of_analysts, delay_creator, image_classifier, maximal_size=10000000):
        self.delay_creator = delay_creator
        self.current_delay_position = 0
        self.maximal_size = maximal_size
        self.delayed_classifications = {}
        self.number_of_analysts = number_of_analysts
        self.busy_analysts = 0
        self.job_queue = []
        self.image_classifier = image_classifier

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
        if not len(self.job_queue) == 0:
            sample_to_return = self.job_queue[0]
            self.job_queue = self.job_queue[1:]
            self.__start_processing_of_job(sample_to_return)


    def advance_time(self):
        self.current_delay_position = (self.current_delay_position + 1) % self.maximal_size
        if self.current_delay_position in self.delayed_classifications:
            elements_to_add = self.delayed_classifications.get(self.current_delay_position)
            for element in elements_to_add:
                self.free_job()
                if self.image_classifier != None:
                    self.image_classifier.add_image_sample(element[0], element[1])
            del self.delayed_classifications[self.current_delay_position]

    def distance_to_samples_in_work(self, sample):
        samples_in_queue = [sample_file for (sample_file, y) in self.job_queue]
        for key in self.delayed_classifications.keys():
            processed_elements = self.delayed_classifications.get(key)
            for element in processed_elements:
                samples_in_queue.append(element[0])

        clean_distance = self.image_classifier.distance_from_sample(samples_in_queue, sample)
        if(clean_distance != None):
            return clean_distance

        return 10.0

