import numpy as np
import cv2
import os
import cPickle
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import os

data_path = os.path.join('dataset','ImageCLEFmed2009_train.02')


image_paths = []
sift = cv2.xfeatures2d.SIFT_create()
for image_path in os.listdir(data_path):
            
            #count+=1
	    
            image = cv2.imread(os.path.join(data_path,image_path))

            try:
                
                kp, des = sift.detectAndCompute(image,None)
                if des is not None:
                    
                    image_paths.append(os.path.join(data_path,image_path))
		    print os.path.join(data_path,image_path)
		    exit(0)

            except Exception as e:

                print e

print len(image_paths)
ipath_cache_file = os.path.join('cache', 'paths.pkl')
print('Saving to: ' + ipath_cache_file)
with open(ipath_cache_file, 'wb') as f:
    cPickle.dump(image_paths, f)
print 'Done!'
