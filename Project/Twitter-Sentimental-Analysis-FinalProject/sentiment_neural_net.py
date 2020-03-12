

import tensorflow as tf
import pickle
import datetime
tf.compat.v1.disable_eager_execution()

with tf.compat.v1.name_scope('input'):
    x = tf.compat.v1.placeholder('float',name="x-input")
    y = tf.compat.v1.placeholder('float',name="y-input")

batch_size = 1000
num_epochs = 10

logs_path = '/tmp/tensorflow_logs/output/'

summary_writer = tf.summary.create_file_writer(logs_path)

def load_details():
    with open('process/data_details.pkl', 'rb') as details:
        det = pickle.load(details)
        return det

line_sizes = load_details()

# Creates the neural network model
def ff_neural_net(input_data):
    neurons_hl1 = 1500
    neurons_hl2 = 1500
    neurons_hl3 = 1500

    output_neurons = 2

    with tf.compat.v1.name_scope('Layer1'):
        with tf.compat.v1.name_scope('weights'):
            l1_weight = tf.Variable(tf.compat.v1.random_normal([line_sizes['dict'], neurons_hl1]), name='w1')
        with tf.compat.v1.name_scope('biases'):
            l1_bias = tf.Variable(tf.compat.v1.random_normal([neurons_hl1]), name='b1')

    with tf.compat.v1.name_scope('Layer2'):
        with tf.compat.v1.name_scope('weights'):
            l2_weight = tf.Variable(tf.compat.v1.random_normal([neurons_hl1, neurons_hl2]), name='w2')
        with tf.compat.v1.name_scope('biases'):
            l2_bias = tf.Variable(tf.compat.v1.random_normal([neurons_hl2]), name='b2')

    with tf.compat.v1.name_scope('Layer3'):
        with tf.compat.v1.name_scope('weights'):
            l3_weight = tf.Variable(tf.compat.v1.random_normal([neurons_hl2, neurons_hl3]), name='w3')
        with tf.compat.v1.name_scope('biases'):
            l3_bias = tf.Variable(tf.compat.v1.random_normal([neurons_hl3]), name='b3')

    with tf.compat.v1.name_scope('OutputLayer'):
        with tf.compat.v1.name_scope('weights'):
            output_weight = tf.Variable(tf.compat.v1.random_normal([neurons_hl3, output_neurons]), name='wo')
        with tf.compat.v1.name_scope('biases'):
            output_bias = tf.Variable(tf.compat.v1.random_normal([output_neurons]), name='bo')

    with tf.compat.v1.name_scope("Layer1Processing"):
        with tf.compat.v1.name_scope('W1_plus_b'):
            l1 = tf.add(tf.matmul(input_data, l1_weight), l1_bias)
            l1 = tf.nn.relu(l1)
            tf.summary.histogram("relu1", l1)

    with tf.compat.v1.name_scope("Layer2Processing"):
        with tf.compat.v1.name_scope('W2_plus_b'):
            l2 = tf.add(tf.matmul(l1, l2_weight), l2_bias)
            l2 = tf.nn.relu(l2)
            tf.summary.histogram("relu2", l2)


    with tf.compat.v1.name_scope("Layer3Processing"):
        with tf.compat.v1.name_scope('W3_plus_b'):
            l3 = tf.add(tf.matmul(l2, l3_weight), l3_bias)
            l3 = tf.nn.relu(l3)
            tf.summary.histogram("relu3", l3)


    with tf.compat.v1.name_scope("OutputProcessing"):
        output = tf.matmul(l3, output_weight) + output_bias

    return output


def training(in_placeholder):
    nn_output = ff_neural_net(in_placeholder)
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.name_scope('total'):
        # We are using cross entropy to calculate the cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_output, labels=y))

    with tf.compat.v1.name_scope('train'):
        # and Gradient Descent to reduce the cost
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # A TensorFLow session is created that will actually run the previously defined graph
    with tf.compat.v1.Session() as sess:
        # saver = tf.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        writer = tf.summary.create_file_writer("/tmp/mylogs/sslx")

        for epoch in range(num_epochs):
            epoch_loss = 0
            buffer_train = []
            buffer_label = []
            with open('process/train_hot_vectors.pickle', 'rb') as train_hot_vec:
                for i in range(line_sizes['train']):
                    hot_vector_line = pickle.load(train_hot_vec)
                    buffer_train.append(hot_vector_line[0])
                    buffer_label.append(hot_vector_line[1])

                    # print('Bla:' + str(buffer_label))
                    # print(len(buffer_train))


                    if len(buffer_train) >= batch_size:

                        _, cost_iter = sess.run([optimizer, cost],
                                                feed_dict={in_placeholder: buffer_train, y: buffer_label})
                        epoch_loss += cost_iter
                        buffer_train = []
                        buffer_label = []

                # with tf.compat.v1.name_scope('metrics'):
                #   with writer.as_default():
                #         print("nisha")
                #         f=tf.Variable(epoch_loss)
                #         # Create a summary to monitor cost tensor
                #         tf.summary.scalar("loss", f,step=epoch)
                #         # Create a summary to monitor accuracy tensor
                #         tf.summary.scalar("accuracy", tf.reduce_mean(tf.cast(tf.equal(tf.argmax(nn_output, 1), tf.argmax(y, 1)), 'float')),step=epoch)
                #         # # Create summaries to visualize weights
                #         # for var in tf.compat.v1.trainable_variables():
                #         #     tf.summary.histogram(var.name, var,step=epoch)
                #         # # Merge all summaries into a single op
                #         all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
                #         writer_flush = writer.flush()
                #
                #         print(epoch)
                #
                #         sess.run(tf.compat.v1.global_variables_initializer())
                #         sess.run([writer.init()])
                #         sess.run([all_summary_ops, writer_flush],
                #                         feed_dict={in_placeholder: buffer_train, y: buffer_label})

                print('Epoch {} completed. Total loss: {}'.format(epoch+1, epoch_loss))

        with tf.compat.v1.name_scope('accuracy'):
            with tf.compat.v1.name_scope('correct_prediction'):
                correct = tf.equal(tf.argmax(nn_output, 1), tf.argmax(y, 1))
            with tf.compat.v1.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        with open('process/test_hot_vectors.pickle', 'rb') as train_hot_vec:
            buffer_test = []
            buffer_test_label = []
            for i in range(line_sizes['test']):
                test_hot_vector_line = pickle.load(train_hot_vec)
                buffer_test.append(test_hot_vector_line[0])
                buffer_test_label.append(test_hot_vector_line[1])

        summary_writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())
        summary_writer.close()

        # the accuracy is the percentage of hits
        print('Accuracy using test dataset: {}'
              .format(accuracy.eval({in_placeholder: buffer_test, y: buffer_test_label})))
        # saver = tf.train.Saver()
        saver.save(sess, "process/model.ckpt")

training(x)


def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + logs_path)
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()





