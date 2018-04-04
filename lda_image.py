import numpy as np
import lda
from step1_image import multiModal_features
import os
import cPickle
import cv2

def euclidean(x,y):

    return np.sqrt(np.sum(np.square(x-y)))

class lda_image():

    def __init__(self,data_path):

        # self.histogram = multiModal_features(data_path).vbow.astype(np.int64)
        self.num_topics = 100
        # print self.histogram.shape
        # print np.sum(self.histogram)
        # exit(0)
        #sum = np.sum(self.histogram,axis=0)
        #print sum
        #print np.sum(sum==0)
        #exit(0)
        #print self.histogram.shape
        #self.lda_features = self.get_lda()
        self.lda_model = self.get_lda()
        self.doc_topics = self.lda_model.doc_topic_
        # print self.doc_topics.shape
        # self.query(200)

    def get_lda(self):
        
        lda_cache_file = os.path.join('cache', 'lda.pkl')

        if os.path.isfile(lda_cache_file):
            print('Loading lda object from : ' + lda_cache_file)
            with open(lda_cache_file, 'rb') as f:
                model = cPickle.load(f)
            print 'Done!'
            return model
        else:
            
            model = lda.LDA(self.num_topics, n_iter=500, random_state=1)
            print 'fitting lda...'
            model.fit(self.histogram)
            print 'Done!'
            print('Saving lda object to: ' + lda_cache_file)
            with open(lda_cache_file, 'wb') as f:
                cPickle.dump(model, f)
            print 'Done!'
        
            return model
       
       
        # print self.lda_model.topic_word_[0]
        

    def query(self,idx):
        
        ipath_cache_file = os.path.join('cache', 'paths.pkl')

        if os.path.isfile(ipath_cache_file):
            print('Loading image paths from : ' + ipath_cache_file)
            with open(ipath_cache_file, 'rb') as f:
                ipath = cPickle.load(f)
            print 'Done!'

        img  = cv2.imread(ipath[idx])
        cv2.imwrite(os.path.join('query_demo','query.jpg'),img)
        query_topics = self.lda_model.transform(np.expand_dims(self.histogram[idx],axis = 0),500)

        print query_topics.shape

        result = []

        for i,doc_topics in enumerate(self.doc_topics):

            doc_topics = doc_topics.reshape(1,self.num_topics)
            dis = euclidean(doc_topics,query_topics)
            if(len(result)<4):
                result.append((dis,i))
                result = sorted(result)
            else:
                


                if(dis<result[3][0]):
                    result[3]=(dis,i)

                
                result = sorted(result)
                
                #print result

                # if(i==20):
                #     break
        
        for i,x in enumerate(result):

            img = cv2.imread(ipath[x[1]])
            cv2.imwrite(os.path.join('query_demo',str(i)+'.jpg'),img)
            print x

    def query2(self):

        ipath_cache_file = os.path.join('cache', 'paths.pkl')

        if os.path.isfile(ipath_cache_file):
            # print('Loading image paths from : ' + ipath_cache_file)
            with open(ipath_cache_file, 'rb') as f:
                ipath = cPickle.load(f)
            # print 'Done!'
        
        idx = np.random.randint(0,len(self.doc_topics))
        img  = cv2.imread(ipath[idx])
        # query_topics = self.lda_model.transform(np.expand_dims(self.histogram[idx],axis = 0),500)
        query_topics = self.doc_topics[idx]
        result = []
        for i,doc_topics in enumerate(self.doc_topics):

            doc_topics = doc_topics.reshape(1,self.num_topics)
            dis = euclidean(doc_topics,query_topics)
            result.append((dis,ipath[i][:-4]))

        result = sorted(result)
        # print result
        return result,ipath[idx][:-4]

        

    
    





if __name__ =='__main__' :

    data_path = os.path.join('dataset','ImageCLEFmed2009_train.02')
    model = lda_image(data_path)
    print model.query2()
    # print 'enter query image index'
    # while(1):
    #     k = raw_input()
    #     if (k=='exit'):
    #         break
    #     idx = int(k)
    #     model.query(idx)