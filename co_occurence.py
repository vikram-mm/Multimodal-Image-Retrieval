from step1_text import get_textual_features
from lda_image import lda_image
import os
import cPickle
import numpy as np

def get_vistex():

    vistex = []
    data_path = os.path.join('dataset','ImageCLEFmed2009_train.02')
    model = lda_image(data_path)
    textual_data,textual_vocab = get_textual_features()
    print len(textual_data)
    ipath_cache_file = os.path.join('cache', 'paths.pkl')

    if os.path.isfile(ipath_cache_file):
        print('Loading image paths from : ' + ipath_cache_file)
        with open(ipath_cache_file, 'rb') as f:
            ipath = cPickle.load(f)
        print 'Done!'

    for i,doc_topics in enumerate(model.doc_topics):

        # print doc_topics
        # exit(0)

        # if(i==10):
        #     break
        # print len(vistex)
        try:
            textual_words = textual_data[ipath[i][:-4]]
        except:
            # print 'image not found in csv!'
            continue
        feature = np.zeros((textual_vocab,model.num_topics))

        # for t in textual_words:
        #     for d,topic in enumerate(doc_topics):
        #         feature[t,d] = topic
        feature[textual_words,:] = doc_topics

        vistex.append(feature)
    
    return np.stack(vistex)





if __name__ == '__main__':

    # np.set_printoptions(threshold='nan')
    # ipath_cache_file = os.path.join('cache', 'paths.pkl')

    # if os.path.isfile(ipath_cache_file):
    #     print('Loading image paths from : ' + ipath_cache_file)
    #     with open(ipath_cache_file, 'rb') as f:
    #         ipath = cPickle.load(f)
    #     print 'Done!'

  

    x = get_vistex()
    # print x
    print x.shape