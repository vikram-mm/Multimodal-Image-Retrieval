import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans


import os

def tf_kmeans(full_data_x):
        num_steps = 50 # Total steps to train
        batch_size = 1024 # The number of samples per batch
        k = 3000 # The number of clusters
        num_features = 128 # Each image is 28x28 pixels
        full_data_x = full_data_x


        # Input images
        X = tf.placeholder(tf.float32, shape=[None, num_features])


        # K-Means Parameters
        kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

        # Build KMeans graph
        training_graph = kmeans.training_graph()

        if len(training_graph) > 6: # Tensorflow 1.4+
            (all_scores, cluster_idx, scores, cluster_centers_initialized,
            cluster_centers_var, init_op, train_op) = training_graph
        else:
            (all_scores, cluster_idx, scores, cluster_centers_initialized,
            init_op, train_op) = training_graph

        cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
        avg_distance = tf.reduce_mean(scores)

        # Initialize the variables (i.e. assign their default value)
        init_vars = tf.global_variables_initializer()

        # Start TensorFlow session
        sess = tf.Session()

        # Run the initializer
        sess.run(init_vars, feed_dict={X: full_data_x})
        sess.run(init_op, feed_dict={X: full_data_x})

        # cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)

        # Training
        for i in range(1, num_steps + 1):
             _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
            if i % 10 == 0 or i == 1:
                print("Step %i, Avg Distance: %f" % (i, d))

        return idx


        