import tensorflow as tf
import numpy as np

from pipeline.delay.UniformDelay import UniformDelay
from pipeline.environments.ContinuousOmniglotMulticlassGeneratorReallyRandomLabels import \
    ContinuousOmniglotMulticlassGeneratorReallyRandomLabels
from pipeline.environments.ContinuousOmniglotMulticlassGeneratorReallyRandomLabelsRepeatedImages import \
    ContinuousOmniglotMulticlassGeneratorReallyRandomLabelsRepeatedImages
from pipeline.environments.ContinuousOmniglotMulticlassNoReadingGenerator import \
    ContinuousOmniglotMulticlassNoReadingGenerator
from pipeline.learningSets.OnlineUpdateInsertions import OnlineUpdateInsertions
from pipeline.management.AnalystManagement import AnalystManagement
from pipeline.rlAgent.BulksTwoPartOMNIGLOTAgentFullImage import BulksTwoPartOMNIGLOTAgentFullImage
from pipeline.rlAgent.StorageBasedImageClassifier import StorageBasedImageClassifier
from pipeline.rlAgent.StorageBasedImageClassifierNoNorm import StorageBasedImageClassifierNoNorm
from pipeline.rlAgent.StorageBasedImageClassifierNormalized import StorageBasedImageClassifierNormalized
from pipeline.rlAgent.SuggestedGeneralLSTMPolicy import SuggestedGeneralLSTMPolicy
from pipeline.rlAgent.SuggestedGeneralLSTMPolicyMNIH import SuggestedGeneralLSTMPolicyMNIH
from pipeline.rlAgent.SuggestedGeneralLSTMPolicyPreweights import SuggestedGeneralLSTMPolicyPreweights
from pipeline.rlAgent.SuggestedGeneralLSTMPolicyWOP import SuggestedGeneralLSTMPolicyWOP

np.set_printoptions(threshold='nan')
import time
import sys
import os

from pipeline.analysts.NoUpdateRegularAnalysts import NoUpdateRegularAnalysts
from pipeline.analysts.RegularAnalysts import RegularAnalysts
from pipeline.delay.ConstantDelay import ConstantDelay
from pipeline.delay.PoissonDelay import PoissonDelay
from pipeline.environments.ContinuousMalBenFromOmniglotGenerator import ContinuousMalBenFromOmniglotGenerator
from pipeline.environments.ContinuousOmniglotMulticlassGenerator import ContinuousOmniglotMulticlassGenerator
from pipeline.environments.OmniglotGeneratorWithCache import OmniglotGeneratorWithCache
from pipeline.graphs.SampleAverager import SampleAverager
from pipeline.learningSets.DelayClassification import DelayClassification
from pipeline.learningSets.DelayedInsertions import DelayedInsertions
from pipeline.rlAgent.BulksTwoPartOMNIGLOTAgent import BulksTwoPartOMNIGLOTAgent
from pipeline.rlAgent.BulksTwoPartOMNIGLOTAgentClassifier import BulksTwoPartOMNIGLOTAgentClassifier
from pipeline.rlAgent.ContinuousLSTMOnlyOmniglotPaperAgent import ContinuousLSTMOnlyOmniglotPaperAgent
from pipeline.rlAgent.LSTMOnlyOmniglotPaperAgent import LSTMOnlyOmniglotPaperAgent
from pipeline.rlAgent.BulksLSTMOnlyOmniglotPaperAgent import BulksLSTMOnlyOmniglotPaperAgent


# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


##Configurations
experiment_name = "Duh-many"
generator_batch_size = 1
system_batch_size = 1
loaded_characters = 3
total_classes = 30
actions = total_classes + 2
analysts_slots_in_x = 1
nb_samples_per_class = 10
samples_per_batch = loaded_characters * nb_samples_per_class
# samples_per_batch_m1 = samples_per_batch - 1
input_size = 20 * 20
total_input_size = 34
lstm_units = 200
policy_lstm_units = 200
learning_rate = 0.001
gamma = 0.5
e_lambda = 0.7
eps = 0.05
anneal_eps = 0
training_eps = 0.05
number_of_analysts = 3
delay_classification_penalty_base = -0.05
delay_classification_penalty_multiplier = 2
classification_delay_timesteps = 10
wrong_classification_penalty = -1
analyst_load_penalty_multiplier = -0.1
full_analyst_load = 5.0
full_analysts_penalty = -5.0
analyst_delay_param = 9.0
maximal_delay_size = 10000000
classifier_training_batches = 1000
system_training_batches = 3000000
samples_to_average_in_graphs = 50
classifier_training_display_step = 100
statistics_siplay_step = 50
minimal_database_size = 0
max_number_of_delayed = 100
is_training_phase = True

num_of_reward_averaging = 4

print globals()

tf.reset_default_graph()

graph_data_creator = SampleAverager(samples_to_average=samples_to_average_in_graphs, samples_per_batch=samples_per_batch, averages_to_print=1)
rlAgent = SuggestedGeneralLSTMPolicyWOP(batch_size=system_batch_size, samples_per_batch=samples_per_batch, total_input_size=total_input_size,
                                    actions=actions, learning_rate=learning_rate)
image_classifier = StorageBasedImageClassifierNoNorm(total_classes, memory_size=25)
# classifier_sample_manager = OnlineUpdateInsertions(maximal_size=maximal_delay_size)
analyst_expr_manager = DelayedInsertions(samples_per_batch, maximal_size=maximal_delay_size, graph_creator=None)
samples_delay_manager = DelayClassification(max_load=max_number_of_delayed, maximal_size=maximal_delay_size)
delay_creator = UniformDelay(analyst_delay_param)
# job_manager = NoUpdateRegularAnalysts(number_of_analysts, classifier_sample_manager, delay_creator)
job_manager = AnalystManagement(number_of_analysts, maximal_delay_size, delay_creator, image_classifier)
# classifier_sample_manager.set_agents(job_manager)
# classifier_sample_manager.set_classifier(image_classifier)


init = tf.global_variables_initializer()

def benchmark(argv):
    # global current_timestep
    print "Train data " + str(argv[0])
    # print "System Train data " + str(argv[1])
    # print "System Test data: " + str(argv[2])

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

    with tf.Session(config=config) as sess:
        sess.run(init)

        # try to load a saved model
        model_loader = tf.train.Saver()
        current_model_folder = "./models/backup/model-" + experiment_name
        if (os.path.exists(current_model_folder)):
            print("Loading pre calculaated model")
            model_loader.restore(sess, current_model_folder + "/model.ckpt")
            print("v1 : %s" % rlAgent.policy_biases['out'].eval()[0])
        else:
            print("Creating the folder for the model to be stored in")
            os.makedirs(current_model_folder)

        generator = ContinuousOmniglotMulticlassGeneratorReallyRandomLabels(data_folder=argv[0],
                                                          batch_size=generator_batch_size, nb_samples=loaded_characters,
                                                          nb_samples_per_class=nb_samples_per_class,
                                                          max_rotation=0., max_shift=0., max_iter=None, only_labels_and_images=True)

        t0 = time.time()

        print("Starting real training")
        # generator = ContinuousOmniglotMulticlassGenerator(data_folder=argv[0],
        #                                                   batch_size=generator_batch_size, nb_samples=loaded_characters,
        #                                                   nb_samples_per_class=nb_samples_per_class,
        #                                                   max_rotation=0., max_shift=0., max_iter=None)
        # sample_buffer_m1 = np.zeros(total_input_size)
        # y_m1 = 0
        first_iteration = True

        global eps
        global is_training_phase

        # classifier_state = ([classifier_state[0][0,:]], [classifier_state[1][0,:]])  # Reset the recurrent layer's hidden state
        policy_online_state = (np.zeros([system_batch_size, policy_lstm_units]),
                 np.zeros([system_batch_size, policy_lstm_units]))  # Reset the policy layer's hidden state

        policy_training_state = (np.zeros([system_batch_size, policy_lstm_units]),
                                 np.zeros(
                                     [system_batch_size, policy_lstm_units]))  # Reset the policy layer's hidden state

        accuracy_agg = []
        confusion_matrix = np.zeros(6)
        sample_per_batch_agg = []
        reward_sampling = []
        loss_agg = []

        r_m1 = None
        a_m1 = None
        x_m1 = None

        for i, image_and_labels in generator:
            analyst_load = []
            classification_requests = np.zeros(samples_per_batch)

            for batch in range(generator_batch_size):
                sample_counter = 0
                labels, image_files = zip(*image_and_labels[batch])
                samples_length = len(image_files)
                if ((i % samples_to_average_in_graphs == 0) and (batch == 0)):
                    print("Starting policy training on step " + str(i) + ", over data with length: " + str(samples_length))
                    if(is_training_phase):
                        print("Saving most recent model")
                        save_path = model_loader.save(sess, current_model_folder + "/model.ckpt")
                        print("Model saved in path: %s" % save_path)
                        print("v1 : %s" % rlAgent.policy_biases['out'].eval()[0])


                timeseries = np.array(image_files)
                batch_y = np.array([int(float(x)) for x in labels])

                sample = 0

                # Experiment 1 - try to predict bulks
                one_hot_y = np.zeros((batch_y.shape[0], total_classes))
                one_hot_y[np.arange(batch_y.shape[0]), batch_y] = 1
                # Eperiment 1

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
                        # print("starting evaluation on historic data with hash " + str(hash(str(x_sample_buffer))))
                    else:
                        classification_logits_sample = image_classifier.classify(timeseries[sample])
                        y = int(batch_y[sample])
                        x_sample_buffer = image_classifier.load_representing_image(timeseries[sample])
                        sample_file = timeseries[sample]
                        # classification_logits_sample = classification_logits[sample]
                        # print("starting evaluation on new data with hash " + str(hash(str(x_sample_buffer))))

                    accuracy_agg.append(np.argmax(classification_logits_sample) == y)
                    sample_counter += 1

                    # plt.imshow(np.reshape(x_sample_buffer, [20,20]), cmap='gray')
                    # plt.show()

                    sample_for_policy = np.concatenate(
                        ([samples_delay_manager.get_load()], [delay_penalty], [job_manager.analysts_load()], [job_manager.distance_to_samples_in_work(sample_file)], np.reshape(classification_logits_sample, -1)))
                    # print("buffer " + str(hash(str(x_sample_buffer))) + " transformed into " + str(hash(str(sample_for_policy))))
                    qp1, a, new_online_state = sess.run([rlAgent.policy_logits, rlAgent.predicted_action, rlAgent.policy_rnn_state],
                                                        feed_dict={rlAgent.X: np.reshape(sample_for_policy, [system_batch_size, 1, -1]),
                                                                   rlAgent.policy_state_in: policy_online_state, rlAgent.timeseries_length: 1})

                    policy_online_state = new_online_state
                    a = a[0]
                    qp1 = qp1[0]
                    # print("policy suggested action " + str(a) + " for data with hash " + str(hash(str(sample_for_policy))) + ", load is " + str(job_manager.analysts_load()))
                    analyst_load.append(job_manager.analysts_load())


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

                    if (image_classifier.database_size() < minimal_database_size):
                        print("database has not reached size of {} - requesting label".format(minimal_database_size))
                        a = total_classes

                    # print("action which will be taken is: " + str(a) + " for data with hash " + str(
                    #     hash(str(sample_for_policy))) + ", load is " + str(job_manager.analysts_load()))

                    reward = 1
                    if (a == total_classes + 1):  # Delay in classification
                        if(samples_delay_manager.get_number_of_delayed() >= max_number_of_delayed ):
                            reward = wrong_classification_penalty
                            confusion_matrix[4] += 1
                        else:
                            reward = delay_penalty
                            delay_penalty *= delay_classification_penalty_multiplier
                            # print(
                            # "delaying item " + str(hash(str(x_sample_buffer))) + " for: " + str(
                            #     classification_delay_timesteps) + ", will cause penalty of: " + str(reward))
                            samples_delay_manager.add_item((delay_penalty, x_sample_buffer, y, sample_file), delay=classification_delay_timesteps)
                            confusion_matrix[2] += 1
                    elif (a == total_classes):  # Asked for classification
                        if(job_manager.analysts_load() < full_analyst_load):
                            job_manager.add_classification_job((sample_file, y))
                            reward = analyst_load_penalty_multiplier * job_manager.analysts_load()
                            # classification_requests[sample] = classification_requests[sample] + 1
                            confusion_matrix[3] += 1
                        else:
                            reward = wrong_classification_penalty
                            confusion_matrix[5] += 1
                    else:
                        # print "Agent tried to predict"
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

                    # Training occurs indeoendant of buffer filling - now we have two type of training
                    # We train the classifier and the policy independently
                    # Train the policy
                    if( (analyst_expr_manager.number_of_batches() >= system_batch_size) and is_training_phase):
                        # print "Starting the training - batch " + str(i) + ", at timestep: " + str(
                        #     analyst_expr_manager.get_current_position())
                        expr_matrix = np.reshape(analyst_expr_manager.create_batch(system_batch_size), [-1, 4])
                        x_matrix = np.reshape(np.vstack(expr_matrix[:, 0]),
                                              [system_batch_size, samples_per_batch, total_input_size])
                        action_matrix = np.reshape(expr_matrix[:, 1], [-1, samples_per_batch])
                        q_calc_expr = expr_matrix[:, 2] + gamma * expr_matrix[:, 3]
                        q_calc_expr = np.reshape(q_calc_expr, [-1, samples_per_batch])

                        state_t, _, loss = sess.run([rlAgent.policy_rnn_state, rlAgent.updateModel, rlAgent.loss],
                                              feed_dict={rlAgent.X: x_matrix, rlAgent.actions: action_matrix,
                                                         rlAgent.Q_calculation: q_calc_expr, rlAgent.policy_state_in: policy_training_state})
                        # policy_training_state = state_t
                        loss_agg.append(loss)
                        policy_training_state = policy_online_state


                    if(a != total_classes + 1): #advance if not delayed
                        job_manager.advance_time()
                        if(not samples_delay_manager.is_waiting_sample()):
                            samples_delay_manager.advance_state()
                    if(not is_delayed_sample):
                        sample += 1

                sample_per_batch_agg.append(sample_counter)


            #Statistics
            # graph_data_creator.add_classification_request_sample(classification_requests / system_batch_size)
            # print "Total request overall: " + str(classification_requests / batch_size)
            graph_data_creator.add_analyst_load_sample(np.average(analyst_load))
            # print "average analyst load: " + str(np.average(analyst_load))
            # graph_data_creator.add_confusion_matrix_sample(np.average(np.transpose(np.reshape(confusion_matrix, [-1, 8])), 1))
            # print "confusion matrix: " + str(np.average(np.transpose(np.reshape(confusion_matrix, [-1, 6])), 1)/batch_size)
            graph_data_creator.add_score_sample(np.sum(reward_sampling))
            reward_sampling = []

            if(i % statistics_siplay_step == 0 and i!=0):
                print "Batch " + str(i) + " finished after " + str(time.time() - t0) + " seconds"
                print("averaged accuracy since last print is " + str(np.average(accuracy_agg)))
                if(is_training_phase):
                    print("averaged loss since last print " + str(np.average(loss_agg)))
                print("confusion matrix: " + str(confusion_matrix / (generator_batch_size * statistics_siplay_step)))
                print("Average number of samples: new: {}, total: {}".format(samples_per_batch, np.average(sample_per_batch_agg)))

                confusion_matrix = np.zeros(6)
                accuracy_agg = []
                sample_per_batch_agg = []
                loss_agg = []


            if(i < anneal_eps):
                eps = training_eps + (anneal_eps - i)/(float(anneal_eps))
            elif (i < system_training_batches):
                eps = training_eps
            else:
                eps = 0.0 # test
                is_training_phase = False

            if i == system_training_batches*2:
                print("bye bye")
                exit(1)

        print("Finished full system training, it took " + str(time.time() - t0) + " seconds")

        # print ("Chaning source folder to test: " + argv[1])
        # generator = ContinuousOmniglotMulticlassGenerator(data_folder=argv[2],
        #                                                   batch_size=batch_size, nb_samples=loaded_characters,
        #                                                   nb_samples_per_class=nb_samples_per_class,
        #                                                   max_rotation=0., max_shift=0., max_iter=None)

        eps = 0.0
        print globals()
        t0 = time.time()



if __name__ == '__main__':
    benchmark(sys.argv[1:])