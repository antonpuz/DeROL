import numpy as np


class DelayClassification:
    def __init__(self, max_load, maximal_size=100000):
        self.maximal_size = maximal_size
        self.current_position = 0
        self.delayed_inputs = {}
        self.experience_buffer = []
        self.current_batch_buffer = []
        self.agentManager = None
        self.batch_number = 0
        self.delayed_samples = 0
        self.rewards = []
        self.max_load = max_load

    def advance_state(self):
        # Verify no samples were left untouched
        if (len(self.delayed_inputs.get(self.current_position, [])) != 0):
            print("Fatal error in DelayClassification, state advancement is not allowed with unattended samples, " +
                  "there are still " + str(len(self.delayed_inputs.get(self.current_position, []))) + " samples")
            exit(1)

        self.current_position = (self.current_position + 1) % self.maximal_size

    def add_item(self, item_to_add, delay=0):
        # Add element to be written
        position_after_delay = (self.current_position + delay) % self.maximal_size
        to_be_written_after_delay = self.delayed_inputs.get(position_after_delay, [])
        to_be_written_after_delay.append(item_to_add)
        self.delayed_inputs[position_after_delay] = to_be_written_after_delay
        self.delayed_samples += 1


    def get_current_position(self):
        return self.current_position

    def get_number_of_samples_to_handle(self):
        return len(self.delayed_inputs.get(self.current_position, []))

    def is_waiting_sample(self):
        return len(self.delayed_inputs.get(self.current_position, [])) > 0

    def get_buffer_size(self):
        return self.maximal_size

    def get_sample(self):
        if (len(self.delayed_inputs.get(self.current_position, [])) == 0):
            print("in DelayClassification, asked for samples with empty buffer, returning None")
            return None
        else:
            to_be_written_after_delay = self.delayed_inputs.get(self.current_position, [])
            to_return = to_be_written_after_delay[0]
            to_be_written_after_delay = to_be_written_after_delay[1:]
            self.delayed_inputs[self.current_position] = to_be_written_after_delay
            self.delayed_samples -= 1
            return to_return

    def get_up_to_samples(self, number_of_samples):
        if (len(self.delayed_inputs.get(self.current_position, [])) == 0):
            print("in DelayClassification, asked for samples with empty buffer, returning None")
            return None
        else:
            to_be_written_after_delay = self.delayed_inputs.get(self.current_position, [])
            to_return = to_be_written_after_delay[0:number_of_samples]
            to_be_written_after_delay = to_be_written_after_delay[number_of_samples:]
            self.delayed_inputs[self.current_position] = to_be_written_after_delay
            return to_return

    def get_all_samples(self):
        if (len(self.delayed_inputs.get(self.current_position, [])) == 0):
            return []
        else:
            to_return = self.delayed_inputs.get(self.current_position, [])
            self.delayed_inputs[self.current_position] = []
            return to_return

    def get_number_of_delayed(self):
        return self.delayed_samples

    def get_load(self):
        return self.delayed_samples / float(self.max_load)