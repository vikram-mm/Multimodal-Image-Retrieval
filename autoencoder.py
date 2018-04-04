import os
import cPickle
import numpy as np
from co_occurence import *
import tensorflow as tf
import tensorflow.contrib.layers as lays


def log(message,file_path=os.path.join('ae_logs','ae_log.txt')):

	print message
	f1=open(file_path, 'a+')
	f1.write(message)
	f1.close()

def autoencoder(inputs):

	# encoder
	# 272 x 100 x 1  ->  136 x 50 x 32
	# 136 x 50 x 32  ->  68 x 25 x 16
	# 68 x 25 x 16   ->  34 x 13 x 1
	encoder_1 = lays.conv2d(inputs, 32, [2, 2], stride=2, padding='SAME')
	encoder_2 = lays.conv2d(encoder_1, 16, [2, 2], stride=2, padding='SAME')
	compressed = lays.conv2d(encoder_2, 1, [2, 2], stride=2, padding='SAME')
	# print compressed
	# exit(0)
	# decoder
	# 34 x 13 x 1    ->  68 x 25 x 16  
	# 68 x 25 x 16  ->  136 x 50 x 32
	# 136 x 50 x 32  ->  272 x 100 x 1
	decoder_1 = lays.conv2d_transpose(compressed, 16, [2, 2], stride=2, padding='SAME')
	decoder_2 = lays.conv2d_transpose(decoder_1, 32, [2, 2], stride=2, padding='SAME')
	decoder_3 = lays.conv2d_transpose(decoder_2, 1, [2, 2], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
	
	return decoder_3[:,:,0:inputs.get_shape().as_list()[2],:],compressed,encoder_1,encoder_2,decoder_1,decoder_2

if __name__=='__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = ''

	learning_rate = 0.01

	num_epoch = 5
	batch_size = 100
	display_step = 1
	input_size = 50

	X = tf.placeholder(tf.float32, [None, 272, 100, 1])

	reconstruction, compressed, encoder_1, encoder_2, decoder_1, decoder_2 = autoencoder(X)

	loss_op = tf.reduce_mean(tf.square(reconstruction - X))

	tf.summary.scalar('loss',loss_op)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	init = tf.global_variables_initializer()

	data_X = get_vistex()

	indices = np.random.permutation(np.arange(data_X.shape[0]))

	data_X = data_X[indices,:,:]

	tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	merged = tf.summary.merge_all()
	saver = tf.train.Saver()

	with tf.Session() as sess:

	    train_writer = tf.summary.FileWriter("ae_logs/",sess.graph)

	    # Run the initializer
	    sess.run(init)

	    

	    # print 'restoring session'
	    # saver.restore(sess, "ae_logs/save.ckpt")
	    # print 'done loading'
	    # exit(0) 

	    i = 0
	    print 'started training'
	    for epoch in range(num_epoch):
	        for step in range(data_X.shape[0]/batch_size):
	            batch_x = data_X[step*batch_size:(step+1)*batch_size]

	            i+=1

	            batch_x = np.expand_dims(batch_x,-1)
	            # print batch_x.shape

	            # Run optimization op (backprop)
	            _,summary = sess.run([train_op,merged], feed_dict={X: batch_x})
	            train_writer.add_summary(summary, i)
	            
	            if step % display_step == 0:
	                # Calculate batch loss and accuracy
	                loss = sess.run(loss_op, feed_dict={X: batch_x})
	                log("LR : "+str(learning_rate)+" Epoch : " + str(epoch) + " Step " + str(step) + ", Loss= " + \
	                    "{:.5f}".format(loss))

	            
	            if i%20 == 0:

	                print 'saving checkpoint'
	                save_path = saver.save(sess, os.path.join('ae_logs','save.ckpt'))
	                print("Model saved in path: %s" % save_path)              

	    print 'done!'