# recreate the graph and visualize it in Tensorboard
# • HW2 solution: recreate the graph and visualize it in Tensorboard
import numpy as np
import tensorflow as tf
import threading

logs_path = '/tmp/tensorflow_logs/hw2/'

myGraph = tf.Graph()
with myGraph.as_default():
    with tf.compat.v1.Session() as session:
       input_matrix=np.mat([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]);
       a=tf.constant(input_matrix,name="Const")
       b= tf.compat.v1.reduce_prod(a,name="prod_b")
       c=tf.compat.v1.reduce_mean(a,name="Mean")
       d=tf.compat.v1.reduce_sum(a,name="sum_a")
       e=tf.add(b,c,name="add_d")
       f=tf.add(e,d,name="final_add")
       init_op=tf.compat.v1.global_variables_initializer()
       session.run(init_op)
       summary_writer = tf.compat.v1.summary.FileWriter(logs_path, graph=myGraph)
       summary_writer.close()
       session.close()



def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + logs_path)
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

