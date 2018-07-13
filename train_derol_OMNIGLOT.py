import tensorflow as tf
import numpy as np

from analyst_manager.AnalystManagement import AnalystManagement
from classifiers.OMNIGLOTClassifier import OMNIGLOTClassifier
from delay.UniformDelay import UniformDelay
from rl_agents.DeROLAgent import DeROLAgent
from sample_generators.OMNIGLOTGenerator import OMNIGLOTGenerator
from sample_handlers.DelayClassification import DelayClassification
from sample_handlers.ExperimentLogger import ExperimentLogger

np.set_printoptions(threshold='nan')
import time
import sys
import os

##Configurations
experiment_name = "one-shot-OMNIGLOT"
batch_size = 1
loaded_characters = 3
total_classes = 30
actions = total_classes + 2
samples_per_class = 10
samples_per_batch = loaded_characters * samples_per_class
total_input_size = 34
lstm_units = 200
learning_rate = 0.001
gamma = 0.5
eps = 0.05
number_of_analysts = 3
delay_classification_penalty_base = -0.05
delay_classification_penalty_multiplier = 2
classification_delay_timesteps = 10
wrong_classification_penalty = -1
analyst_load_penalty_multiplier = -0.1
full_analyst_load = 5.0
analyst_delay_param = 9.0
training_batches = 600000
statistics_siplay_step = 50
max_number_of_delayed = 100
is_training_phase = True
##End Configurations


tf.reset_default_graph()

rlAgent = DeROLAgent(batch_size=batch_size, samples_per_batch=samples_per_batch, total_input_size=total_input_size,
                                    actions=actions, learning_rate=learning_rate)
image_classifier = OMNIGLOTClassifier(total_classes, memory_size=25)
analyst_expr_manager = ExperimentLogger(samples_per_batch, graph_creator=None)
samples_delay_manager = DelayClassification(max_load=max_number_of_delayed)
delay_creator = UniformDelay(analyst_delay_param)
job_manager = AnalystManagement(number_of_analysts, delay_creator, image_classifier)

init = tf.global_variables_initializer()

def benchmark(argv):
    print "Starting Training from data: " + str(argv[0])

    with tf.Session() as sess:
        sess.run(init)

        # try to load a saved model
        model_loader = tf.train.Saver()
        current_model_folder = "./trained_models/backup/model-" + experiment_name
        if (os.path.exists(current_model_folder)):
            print("Loading pre calculaated model")
            model_loader.restore(sess, current_model_folder + "/model.ckpt")
            print("v1 : %s" % rlAgent.policy_biases['out'].eval()[0])
        else:
            print("Creating the folder for the model to be stored in")
            os.makedirs(current_model_folder)

        generator = OMNIGLOTGenerator(data_folder=argv[0], batch_size=batch_size, samples_per_class=samples_per_class,
                        classes=loaded_characters, only_labels_and_images=True)

        t0 = time.time()

        print("Starting real training")

        global eps
        global is_training_phase

        policy_online_state = (np.zeros([batch_size, lstm_units]),
                                np.zeros([batch_size, lstm_units]))  # Reset the policy layer's hidden state

        policy_training_state = (np.zeros([batch_size, lstm_units]),
                                 np.zeros([batch_size, lstm_units]))  # Reset the policy layer's hidden state

        accuracy_agg = []
        confusion_matrix = np.zeros(6)
        sample_per_batch_agg = []
        reward_agg = []
        loss_agg = []

        r_m1 = None
        a_m1 = None
        x_m1 = None

        for i, image_and_labels in generator:

            for batch in range(batch_size):
                sample_counter = 0
                labels, image_files = zip(*image_and_labels[batch])
                samples_length = len(image_files)
                if ((i % statistics_siplay_step == 0) and (batch == 0)):
                    print("Starting policy training on step " + str(i) + ", over data with length: " + str(samples_length))
                    if(is_training_phase):
                        print("Saving most recent model")
                        save_path = model_loader.save(sess, current_model_folder + "/model.ckpt")
                        print("Model saved in path: %s" % save_path)
                        print("v1 : %s" % rlAgent.policy_biases['out'].eval()[0])


                timeseries = np.array(image_files)
                batch_y = np.array([int(float(x)) for x in labels])

                sample = 0
                reward_sampling = []

                while sample < samples_length:

                    x_sample_buffer = None
                    classification_logits_sample = None
                    sample_file = None
                    is_delayed_sample = False
                    delay_penalty = delay_classification_penalty_base
                    y = -1

                    if(samples_delay_manager.is_waiting_sample()):
                        x_sample_buffer_tuple = samples_delay_manager.get_sample()
                        delay_penalty = x_sample_buffer_tuple[0]
                        x_sample_buffer = x_sample_buffer_tuple[1]
                        sample_file = x_sample_buffer_tuple[3]
                        classification_logits_sample = image_classifier.classify(sample_file)
                        y = x_sample_buffer_tuple[2]
                        is_delayed_sample = True
                    else:
                        classification_logits_sample = image_classifier.classify(timeseries[sample])
                        y = int(batch_y[sample])
                        x_sample_buffer = image_classifier.load_representing_image(timeseries[sample])
                        sample_file = timeseries[sample]

                    accuracy_agg.append(np.argmax(classification_logits_sample) == y)
                    sample_counter += 1

                    sample_for_policy = np.concatenate(
                        ([samples_delay_manager.get_load()], [delay_penalty], [job_manager.analysts_load()], [job_manager.distance_to_samples_in_work(sample_file)], np.reshape(classification_logits_sample, -1)))
                    qp1, a, new_online_state = sess.run([rlAgent.policy_logits, rlAgent.predicted_action, rlAgent.policy_rnn_state],
                                                        feed_dict={rlAgent.X: np.reshape(sample_for_policy, [batch_size, 1, -1]),
                                                                   rlAgent.policy_state_in: policy_online_state, rlAgent.timeseries_length: 1})

                    policy_online_state = new_online_state
                    a = a[0]
                    qp1 = qp1[0]

                    rand_action_sample = np.random.rand(1)
                    if rand_action_sample < eps:
                        rand_action_sample = np.random.rand(1)
                        if rand_action_sample < 0.25:
                            a = total_classes
                        elif rand_action_sample < 0.5:
                            a = total_classes + 1
                        elif rand_action_sample < 0.75:
                            a = y
                        else:
                            a = np.random.randint(0, total_classes)

                    reward = 1
                    if (a == total_classes + 1):  # Delay in classification
                        if(samples_delay_manager.get_number_of_delayed() >= max_number_of_delayed ):
                            reward = wrong_classification_penalty
                            confusion_matrix[4] += 1
                        else:
                            reward = delay_penalty
                            delay_penalty *= delay_classification_penalty_multiplier
                            samples_delay_manager.add_item((delay_penalty, x_sample_buffer, y, sample_file), delay=classification_delay_timesteps)
                            confusion_matrix[2] += 1
                    elif (a == total_classes):  # Asked for classification
                        if(job_manager.analysts_load() < full_analyst_load):
                            job_manager.add_classification_job((sample_file, y))
                            reward = analyst_load_penalty_multiplier * job_manager.analysts_load()
                            confusion_matrix[3] += 1
                        else:
                            reward = wrong_classification_penalty
                            confusion_matrix[5] += 1
                    else:
                        if (a != y):
                            reward = wrong_classification_penalty
                            confusion_matrix[1] += 1
                        else:
                            confusion_matrix[0] += 1

                    reward_sampling.append(reward)
                    max_q = max(qp1)
                    if(a_m1 != None and is_training_phase):
                        analyst_expr_manager.add_item(
                            np.array([x_m1, a_m1, r_m1, max_q]), delay=0)
                    x_m1 = sample_for_policy
                    a_m1 = a
                    r_m1 = reward

                    # Train the policy
                    if( (analyst_expr_manager.number_of_batches() >= batch_size) and is_training_phase):
                        expr_matrix = np.reshape(analyst_expr_manager.create_batch(batch_size), [-1, 4])
                        x_matrix = np.reshape(np.vstack(expr_matrix[:, 0]),
                                              [batch_size, samples_per_batch, total_input_size])
                        action_matrix = np.reshape(expr_matrix[:, 1], [-1, samples_per_batch])
                        q_calc_expr = expr_matrix[:, 2] + gamma * expr_matrix[:, 3]
                        q_calc_expr = np.reshape(q_calc_expr, [-1, samples_per_batch])

                        state_t, _, loss = sess.run([rlAgent.policy_rnn_state, rlAgent.updateModel, rlAgent.loss],
                                              feed_dict={rlAgent.X: x_matrix, rlAgent.actions: action_matrix,
                                                         rlAgent.Q_calculation: q_calc_expr, rlAgent.policy_state_in: policy_training_state})
                        loss_agg.append(loss)
                        policy_training_state = policy_online_state


                    if(a != total_classes + 1): #advance timestamp count if sample is not delayed
                        job_manager.advance_time()
                        if(not samples_delay_manager.is_waiting_sample()):
                            samples_delay_manager.advance_state()
                    if(not is_delayed_sample):
                        sample += 1

                reward_agg.append(np.sum(reward_sampling))
                reward_sampling = []
                sample_per_batch_agg.append(sample_counter)


            #Statistics
            if(i % statistics_siplay_step == 0 and i!=0):
                print "Batch " + str(i) + " finished after " + str(time.time() - t0) + " seconds"
                print "Average cycle reward is: " + str(np.sum(reward_agg) / statistics_siplay_step)
                print("averaged accuracy since last print is " + str(np.average(accuracy_agg)))
                if(is_training_phase):
                    print("averaged loss since last print " + str(np.average(loss_agg)))
                print("confusion matrix: " + str(confusion_matrix / (batch_size * statistics_siplay_step)))
                print("Average number of samples: new: {}, total: {}".format(samples_per_batch, np.average(sample_per_batch_agg)))

                #Clear the counters
                confusion_matrix = np.zeros(6)
                accuracy_agg = []
                sample_per_batch_agg = []
                loss_agg = []
                reward_agg = []


            if i == training_batches:
                eps = 0.0
                is_training_phase = False

            if i == training_batches*2:
                print("Finished full system training, execution time " + str(time.time() - t0) + " seconds")
                exit(1)

        print("Finished full system training, it took " + str(time.time() - t0) + " seconds")


if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print("train_derol should be called with image root folder as parameter")
        exit(1)
    benchmark(sys.argv[1:])