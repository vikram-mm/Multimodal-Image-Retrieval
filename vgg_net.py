import os
import cPickle
import numpy as np
from co_occurence import *
import tensorflow as tf
import tensorflow.contrib.layers as lays

from tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim

def infer(inputs, is_training=True):
    inputs = tf.cast(inputs, tf.float32)
    inputs = ((inputs / 255.0)-0.5)*2
    #Use Pretrained Base Model
    with tf.variable_scope("vgg_16"):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            # net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            # net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            # net = slim.max_pool2d(net, [2, 2], scope='pool5')
    #Append fully connected layer
    net1 = slim.flatten(net)
    net = slim.fully_connected(net1, 512,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            scope='finetune/fc1')
    net = slim.fully_connected(net, 2,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            scope='finetune/fc2')
    return net,net1

def losses(logits, labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss
        
def optimize(losses):
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(lr, global_step,
                                             num_iter*decay_per, decay_rate, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(losses, global_step=global_step)#,
                #var_list=slim.get_model_variables("finetune"))
    return train_op

if __name__ == '__main__':
	

	tf.reset_default_graph()

	batch_size=32
	num_epochs=10

	lr = 0.001
	decay_rate=0.1
	decay_per=40 #epoch

	image = tf.placeholder(tf.float32, [None, 272, 100, 3])

	#Create the training graph
	# filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=num_epochs)
	# image, label = read_and_decode(filename_queue)
	prediction,net_x = infer(image)
	# loss = losses(prediction, label)
	# train_op = optimize(loss)

	data_X = get_vistex()
	data_X = np.expand_dims(data_X,-1)

	data_X = np.tile(data_X,(1,1,1,3))

	print data_X.shape
	# exit(0)
	indices = np.random.permutation(np.arange(data_X.shape[0]))

	data_X = data_X[indices,:,:]


	print "Training started"
	with tf.Session() as sess:
	    
	    init_op = tf.group(tf.global_variables_initializer(),
	            tf.local_variables_initializer())
	    restore = slim.assign_from_checkpoint_fn(
	               'vgg_16.ckpt',
	               slim.get_model_variables("vgg_16"))
	    sess.run(init_op)
	    restore(sess)
	    # coord = tf.train.Coordinator()
	    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	    
	    for epoch in range(num_epochs):
	        for step in range(data_X.shape[0]/batch_size):
	            batch_x = data_X[step*batch_size:(step+1)*batch_size]

	           

	            # batch_x = np.expand_dims(batch_x,-1)
	            # print batch_x.shape

	            # Run optimization op (backprop)
	            # _,summary = sess.run([train_op,merged], feed_dict={X: batch_x})
	            # train_writer.add_summary(summary, i)
	            result = sess.run(net_x, feed_dict={image: batch_x})
	            print result
	            break

	        break
	            # exit(0)
	            # if step % display_step == 0:
	            #     # Calculate batch loss and accuracy	                
	            #     log("LR : "+str(learning_rate)+" Epoch : " + str(epoch) + " Step " + str(step))

	    
	    # coord.request_stop()
	    # coord.join(threads)
	    print 'Training Done'
	    saver = tf.train.Saver(slim.get_model_variables())
	    saver.save(sess, 'vgg_logs/model.ckpt')
	    sess.close()