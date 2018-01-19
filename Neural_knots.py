import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import theano
from theano import tensor as T
from Knots import generate_batch

def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))



def architecture_ann(exec_time=10, grid_size=15, layer_num=2, hidden_nodes=300,
                     output_type='regression', learn_rate=1e-1, train_batch=100,
                     valid_size=200, global_epochs=1, descent='gradient'):

    exec_time = time

    errors = np.zeros(exec_time*6 + 1)

    for j in range(global_epochs):
        
        print("Starting global epoch %d" %j)
        
        sess = tf.InteractiveSession()
        knot = tf.placeholder(tf.float32, shape=[None, grid_size**2], name='knot')
        crossings = tf.placeholder(tf.float32, shape=[None,1], name='crossings')    

        if layer_num > 0:
            w_h_1 = init_weights([grid_size**2, num_hidden], 'uniform', xavier_params=(1, num_hidden))
            b_h_1 = init_weights([num_hidden], 'uniform')
            h_1 = tf.nn.relu(tf.matmul(knot, w_h_1) + b_h_1)
        
        if layer_num > 1:
            w_h_2 = init_weights([num_hidden, num_hidden], 'uniform', xavier_params=(1, num_hidden))
            b_h_2 = init_weights([num_hidden], 'uniform')
            h_2 = tf.nn.relu(tf.matmul(h_1, w_h_2) + b_h_2)
            
        if layer_num > 2:
            w_h_3 = init_weights([num_hidden, num_hidden], 'uniform', xavier_params=(1, num_hidden))
            b_h_3 = init_weights([num_hidden], 'uniform')
            h_3 = tf.nn.relu(tf.matmul(h_2, w_h_3) + b_h_3)
            

        #w_o = init_weights([num_hidden, max_cross], 'xavier', xavier_params=(num_hidden, 1))
        w_o = init_weights([num_hidden, 1], 'uniform', xavier_params=(num_hidden, 1))
        
        #b_o = init_weights([max_cross], 'zeros')
        b_o = init_weights([1],'uniform')
        
        if layer_num == 1:
            #crossings_ = tf.nn.softmax(tf.matmul(h_1, w_o) + b_o)
            crossings_ = tf.matmul(h_1, w_o) + b_o
        if layer_num == 2:
            #crossings_ = tf.nn.softmax(tf.matmul(h_2, w_o) + b_o)
            crossings_ = tf.matmul(h_2, w_o) + b_o
        if layer_num == 3:
            #crossings_ = tf.nn.softmax(tf.matmul(h_3, w_o) + b_o)
            crossings_ = tf.matmul(h_3, w_o) + b_o
            
            
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=crossings, logits=crossings_))
        MSE = tf.reduce_mean(tf.squared_difference(crossings, crossings_))

        #correct_prediction = tf.equal(tf.argmax(crossings,1), tf.argmax(crossings_,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(MSE)
        #train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

        sess.run(tf.global_variables_initializer())

        i=0
        num_cycle=0
        t_end = time.time() + 60 * exec_time
        t_start = time.time()
        errors_stack = [0]*10

        while time.time() < t_end:

            batch = generate_batch(grid_size=grid_size, batch_size=train_batch_size, 
                                   representation_knot='matrix', representation_cross='int',
                                   max_cross=max_cross)
            crosses = batch['crossings'].reshape((train_batch_size, 1))
            sess.run(train_step, feed_dict={knot: batch['knot'], crossings: crosses})
            
            if time.time() - t_start > 10*i: 
                
                valid_batch = generate_batch(grid_size=grid_size, batch_size = valid_size, 
                                             representation_knot = 'matrix', representation_cross='int', max_cross=max_cross)
                crosses = valid_batch['crossings'].reshape((valid_size,1))
                errors_stack = errors_stack[1:] + [sess.run(MSE,
                               feed_dict={knot:valid_batch['knot'], crossings:crosses})]
                
                print("after %d seconds, MSE is %.3f "
                      % (int(time.time() - t_start), sum(errors_stack)/10))
                if i > 10:
                    errors[i] = errors[i]*(j/(j+1)) + sum(errors_stack)/(10*(j+1))
                i += 1
        print(" ")

    plt.figure(figsize=(4,2))
    plt.plot(errors)
    plt.xlabel('seconds/10')
    plt.ylabel('MSE')
    plt.show()
    return 


#def architecture_cnn(exec_time=10, grid_size=20, feature_num=5,
                     #layer_num=2):
    
