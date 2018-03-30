
import numpy as np
import cv2
import os
import cPickle
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import os



class multiModal_features():

    def __init__(self, data_path):

         self.data_path = data_path
         self.k = 3000

         self.vbow = self.get_vbow()
    
    def get_vbow(self):

        histogram_cache_file = os.path.join('cache', 'histogram.pkl')

        if os.path.isfile(histogram_cache_file):
            print('Loading histogram from : ' + histogram_cache_file)
            with open(histogram_cache_file, 'rb') as f:
                vbow = cPickle.load(f)
            print 'done'
            return vbow
        else:
            
            print 'creating vbow : '
            vbow = self.createVBOW()
            print('Saving histogram to: ' + histogram_cache_file)
            with open(histogram_cache_file, 'wb') as f:
                cPickle.dump(vbow, f)
            print 'Done!'
            return vbow
        

    
    def createVBOW(self):

        #getting sift descriptors
        sift_cache_file = os.path.join('cache', 'sift_des.pkl')

        if os.path.isfile(sift_cache_file):
            print('Loading sift descriptors from : ' + sift_cache_file)
            with open(sift_cache_file, 'rb') as f:
                self.siftDes = cPickle.load(f)
            print 'Done!'
            print 'length : ',len(self.siftDes)
        else:
            
            print 'Extracting sift descriptors'
            self.siftDes = self.extract_siftDes()
            print 'Done!'
            print 'length : ',len(self.siftDes)
            print('Saving sift descriptors to: ' + sift_cache_file)
            with open(sift_cache_file, 'wb') as f:
                cPickle.dump(self.siftDes, f)
            print 'Done!'
        
        #exit(0)
        #kmeans
        kmeans_cache_file = os.path.join('cache', 'kmeans.pkl')

        if os.path.isfile(kmeans_cache_file):
            print('Loading kmeans object from : ' + kmeans_cache_file)
            with open(kmeans_cache_file, 'rb') as f:
                self.kmeans = cPickle.load(f)
            print 'Done!'
        else:
            
            print 'Performing k means clustering'
            self.kmeans = self.perform_kmeans()
            print 'Done!'
            print('Saving kmeans object to: ' + kmeans_cache_file)
            with open(kmeans_cache_file, 'wb') as f:
                cPickle.dump(self.kmeans, f)
            print 'Done!'
        
        #histogram
        print 'genrating histogram ...'
        vbow = self.generate_histogram()

        return vbow
        
    
    def extract_siftDes(self):

        sift = cv2.xfeatures2d.SIFT_create()
        print (sift)

        siftDes = []
        
        #count  = 0
        for image_path in os.listdir(self.data_path):
            
            #count+=1
            image = cv2.imread(os.path.join(self.data_path,image_path))
            # kp = sift.detect(image,None)
            #print kp
            # visualize = cv2.drawKeypoints(image,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #cv2.imwrite('sift_keypoints'+str(count)+'.jpg',visualize)

            try:
                
                kp, des = sift.detectAndCompute(image,None)
                # print des.shape,des

                # if(count==4):
                #     break
                if des is not None:
                    
                    siftDes.append(des)

            except Exception as e:

                print 'at ',len(siftDes),e
        
        return siftDes
    
    def perform_kmeans(self):

            # kmeans = KMeans(n_clusters=self.k, random_state=0)
            # print 'created kmeans model'
            X = self.siftDes[0]

            print X.shape

            print 'stacking all features..'
            for i in range(1,len(self.siftDes)):
                #print X.shape,self.siftDes[i].shape
                try: #(some are none self.siftDes)
                    X = np.concatenate((X,self.siftDes[i]))
                except:
                    print i
            
            print X.shape

            print 'fitting...'
            idx = tf_kmeans(X,500)

            return idx

    def generate_histogram(self):

        histogram = np.zeros((len(self.siftDes),self.k))
        count = 0
        for i,des in enumerate(self.siftDes):
            # print 'i: ',i
            try:
                labels = self.kmeans[count:count + des.shape[0]]
                # print 'labels ',labels
                for x in labels:
                    # print 'x ',x
                    # print np.sum(histogram)
                    histogram[i,x] = histogram[i,x] + 1
                
                count = count + des.shape[0]
            
            except Exception as e:

                print e
        print count
        print len(self.kmeans)
        if(count==len(self.kmeans)):
            print 'correct'
        return histogram


def tf_kmeans(full_data_x,num_steps=50,k=3000):

    
    batch_size = 1024 
    num_features = full_data_x.shape[1]
    full_data_x = full_data_x


    
    X = tf.placeholder(tf.float32, shape=[None, num_features])


    
    kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
            use_mini_batch=True)

    
    training_graph = kmeans.training_graph()

    if len(training_graph) > 6: 
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
        cluster_centers_var, init_op, train_op) = training_graph
    else:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
        init_op, train_op) = training_graph

    cluster_idx = cluster_idx[0] 
    avg_distance = tf.reduce_mean(scores)

    
    init_vars = tf.global_variables_initializer()

    
    sess = tf.Session()

    
    sess.run(init_vars, feed_dict={X: full_data_x})
    sess.run(init_op, feed_dict={X: full_data_x})

    

    
    for i in range(1, num_steps + 1):
        _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                        feed_dict={X: full_data_x})
        if i % 10 == 0 or i == 1:
            print("Step %i, Avg Distance: %f" % (i, d))

    return idx


        

if __name__ =='__main__' :

    data_path = os.path.join('dataset','ImageCLEFmed2009_train.02')
    multiModal_features(data_path)
