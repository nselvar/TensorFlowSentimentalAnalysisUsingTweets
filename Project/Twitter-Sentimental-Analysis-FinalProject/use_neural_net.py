import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentiment_neural_net import ff_neural_net

tf.compat.v1.disable_eager_execution()

lemm = WordNetLemmatizer()

x = tf.compat.v1.placeholder('float')

# function responsible to receive a sentence, prepare it (tokenizing, lemmatizing and tranforming into the hot vector
def get_sentiment(input_data):
    tf.compat.v1.reset_default_graph()
    pl = tf.compat.v1.placeholder('float')
    nn_output = ff_neural_net(pl)
    saver = tf.compat.v1.train.Saver()
    with open('process/word_dict.pickle', 'rb') as f:
        word_dict = pickle.load(f)

    with tf.compat.v1.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        saver.restore(sess, "process/model.ckpt")
        words = word_tokenize(input_data.lower())
        lemm_words = [lemm.lemmatize(w) for w in words]
        hot_vector = np.zeros(len(word_dict))

        for word in lemm_words:
            if word.lower() in word_dict:
                index_value = word_dict.index(word.lower())
                hot_vector[index_value] += 1

        hot_vector = np.array(list(hot_vector))

        result = (sess.run(tf.argmax(nn_output.eval(feed_dict={pl: [hot_vector]}), 1)))
        print(result)
        if result[0] == 0:
            print('Negative:', input_data)
        elif result[0] == 1:
            print('Positive:', input_data)


# Uncomment the row below to train the model
# training(x)

# call the 'use_neural_network' providing a sentence to check the neural network return
get_sentiment('This is my very first neural network project, I love it.')
get_sentiment("Corona virus is very scary and distrubing")
get_sentiment("I hate you!")
get_sentiment("Game of thrones is intresting and exciting ")

