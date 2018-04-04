import re
import pandas as pd
import numpy as np
from irma_reader import *
from auc_ap import auc
import os
from vgg_net import infer
import tensorflow as tf
from co_occurence import *

def euclidean(x,y):

    return np.sqrt(np.sum(np.square(x-y)))

class vgg_model:

    def __init__(self):

        self.X = tf.placeholder(tf.float32, [None, 272, 100, 3])
        _, self.compressed = infer(self.X)
        saver = tf.train.Saver()
        self.sess= tf.Session()
        print 'restoring session'
        saver.restore(self.sess, "vgg_logs/model.ckpt")
        print 'done loading'
        self.features = self.compute_all_fetures()

        ipath_cache_file = os.path.join('cache', 'paths.pkl')

        if os.path.isfile(ipath_cache_file):
            # print('Loading image paths from : ' + ipath_cache_file)
            with open(ipath_cache_file, 'rb') as f:
                self.ipath = cPickle.load(f)

        

    def compute_all_fetures(self):

        vggfeature_cache_file = os.path.join('cache', 'vgg_features.pkl')

        if os.path.isfile(vggfeature_cache_file):
            print('Loading vgg features from : ' + vggfeature_cache_file)
            with open(vggfeature_cache_file, 'rb') as f:
                vgg_features = cPickle.load(f)
            print 'Done!'
            return vgg_features

        else:

            print 'computing vgg features'
            vistex = np.expand_dims(get_vistex(),-1)
            vistex = np.tile(vistex,(1,1,1,3))
            # features = self.sess.run(self.compressed,feed_dict={self.X:vistex})
            batch_size = 128
            features=[]
            for step in range(vistex.shape[0]/batch_size):
                batch_x = vistex[step*batch_size:(step+1)*batch_size]
                # print batch_x.shape
                batch_features = self.sess.run(self.compressed, feed_dict={self.X: batch_x})
                features.append(batch_features)
                if(step%20==0):
                    print step
            features = np.vstack(features)
            print features.shape
            print 'done'
            # exit(0)
            # features = np.squeeze(features)
            # print features.shape
            # print('Saving vgg features to: ' + vggfeature_cache_file)
            # with open(vggfeature_cache_file, 'wb') as f:
            #     cPickle.dump(features, f)
            # print 'Done!'
            return features

    def query(self):

        idx = np.random.randint(0,len(self.features))
        query_feature = self.features[idx]

        result = []
        for i,feature in enumerate(self.features):

            dis = euclidean(feature,query_feature)
            result.append((dis,self.ipath[i][:-4]))

        result = sorted(result)
        # print result
        return result,self.ipath[idx][:-4]

def evaluate(num_queries=10):

    class_info = pd.read_csv("ImageCLEFmed2009_train_codes.02.csv")
    class_i = np.array(class_info["05_class"])
    print class_i.shape

    class_count = {}

    for c in class_i:
        if not class_count.get(c):
            class_count[c] = 1
        else:
            class_count[c] += 1

    # print class_count
    get_class = {}
    path = "dataset/ImageCLEFmed2009_train.02/"

    for img_id,img_class in zip(class_info["image_id"],class_info["05_class"]):

        get_class[path+str(img_id)] = img_class
    
    # print get_class

    model = vgg_model()

    mAP = 0.0
    count = 0
    
    for i in range(num_queries):

        result,query = model.query()
        try:
            query_class = get_class[query]
        except:
            i = i-1
            continue
        
        count+=1
        print 'query class ',query_class
        print 'total relavant: ', class_count[query_class]
        simplified_result = []
        for x in result:
            # print x[1], get_class[x[1]]
            try:
                if(get_class[x[1]]==query_class):
                    simplified_result.append(1)
                else:
                    simplified_result.append(0)
            except:

                continue

        # print simplified_result
        
        AP = auc(simplified_result,class_count[query_class])
        print 'query ', i, ' AP : ', AP
        mAP+=AP
    
    mAP /= count

    print 'mAP : ', mAP
    return mAP

if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # model = ae_model()
    # print model.query()

    evaluate(1000)