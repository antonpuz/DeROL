from threading import Lock
import numpy as np


class ExperimentLogger:
    def __init__(self, samples_in_batch, maximal_size=10000000, graph_creator=None):
        self.maximal_size = maximal_size
        self.graph_creator = graph_creator
        if maximal_size<=0:
            raise ValueError('maximal value for iterations should be positive, obtained {}'.format(maximal_size))
        self.current_position = 0
        self.delayed_inputs = {}
        self.samples_in_batch = samples_in_batch
        self.experience_buffer = []
        self.current_batch_buffer = []
        self.agentManager = None
        self.lock = Lock()
        self.batch_number = 0
        self.rewards = []

    def set_agents(self, agentManager):
        self.agentManager = agentManager

    def __add_to_internal_current_buffer(self, element_to_add): #private
        self.current_batch_buffer.append(element_to_add)
        if self.graph_creator != None:
            self.rewards.append(element_to_add[2])
        if len(self.current_batch_buffer) == self.samples_in_batch:
            self.experience_buffer.append(self.current_batch_buffer)
            self.current_batch_buffer = []

    def advance_state(self):
        self.current_position = (self.current_position + 1) % self.maximal_size

        # Add elements delayed for now
        if self.current_position in self.delayed_inputs:
            elements_to_add = self.delayed_inputs.get(self.current_position)
            for element in elements_to_add:
                self.__add_to_internal_current_buffer(element)
                if self.agentManager != None:
                    self.agentManager.free_job()
            del self.delayed_inputs[self.current_position]


    def add_item(self, item_to_add, delay=0, update=True):
        if delay == 0:
            self.__add_to_internal_current_buffer(item_to_add)
        else:
            # Add element to be written
            position_after_delay = (self.current_position + delay) % self.maximal_size
            to_be_written_after_delay = self.delayed_inputs.get(position_after_delay, [])
            to_be_written_after_delay.append(item_to_add)
            self.delayed_inputs[position_after_delay] = to_be_written_after_delay

        if update == True:
            self.advance_state()




    def create_batch(self, batch_size):
        if len(self.experience_buffer) < batch_size:
            raise ValueError('Not enough stored batches in experience creator, requested {}, currently buffer of size {}'.format(batch_size, len(self.experience_buffer)))
        if self.graph_creator != None:
            self.graph_creator.add_score_sample(np.average(self.rewards[0:batch_size]))
            self.rewards = self.rewards[batch_size:]

        to_be_returned = self.experience_buffer[0:batch_size]
        self.experience_buffer = self.experience_buffer[batch_size:]
        self.batch_number += 1
        return to_be_returned

    def number_of_batches(self):
        return len(self.experience_buffer)

    def get_current_position(self):
        return self.current_position

    def get_buffer_size(self):
        return self.maximal_size