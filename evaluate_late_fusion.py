import re
import pandas as pd
import numpy as np
from irma_reader import *
from metrics import *
import os
from autoencoder import autoencoder
from co_occurence import *
from step_2_text import *
from step1_text import get_textual_features

def euclidean(x,y):

    return np.sqrt(np.sum(np.square(x-y)))

vistex = get_vistex()



ipath_cache_file = os.path.join('cache', 'paths.pkl')

if os.path.isfile(ipath_cache_file):
    with open(ipath_cache_file, 'rb') as f:
        ipath = cPickle.load(f)

def vistex_query():

    idx = np.random.randint(0,len(vistex))
    query_feature = vistex[idx]

    result = []
    for i,feature in enumerate(vistex):

        dis = euclidean(feature,query_feature)
        result.append((dis,ipath[i][:-4]))

    result = sorted(result)
    return result,ipath[idx][:-4]

def one_hot(textual_features,vocab_size):

    ans = np.zeros((1,vocab_size))
    for j in textual_features:
        ans[0,j] = 1
    return ans
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

    get_class = {}
    path = "dataset/ImageCLEFmed2009_train.02/"

    for img_id,img_class in zip(class_info["image_id"],class_info["05_class"]):

        get_class[path+str(img_id)] = img_class

    textual_data,textual_vocab = get_textual_features()

    mAP = 0.0
    average_ndgc = 0.0
    gmap = 1.0
    count = 0
    average_precision = np.zeros(3)
    
    model = get_svm()
    for i in range(num_queries):

        result,query = vistex_query()
        try:
            query_class = get_class[query]
        except:
            i = i-1
            continue

        textual_features = textual_data[query]
        textual_features = one_hot(textual_features,textual_vocab)
        pred_class = model.predict(textual_features)[0]
        print 'class predicted using textual fetures only - ', pred_class

        for j,x in enumerate(result):
            try:

                if get_class[x[1]] == '\\N':
                    actual_class=0
                else:
                    actual_class = int(get_class[x[1]])
                x = list(x)
                x[0]+= 0.01*((actual_class-pred_class)**2)
                result[j] = tuple(x)
            except Exception as e:

                continue


        result = sorted(result)

        count+=1
        print 'query class ',query_class

        simplified_result = []
        
        for x in result:
            try:
                if(get_class[x[1]]==query_class):
                    simplified_result.append(1)
                else:
                    simplified_result.append(0)
            except:

                continue

        
        AP = auc(simplified_result,class_count[query_class])
        ndgc = calc_ndgc(simplified_result)
        precision = calc_precision(simplified_result)
        print 'query ', i, ' AP : ', AP, 'NDGC : ', ndgc,'p5 : {}, p10 : {}, p20 : {}'.format(precision[0],precision[1],precision[2])
        mAP+=AP
        gmap*=AP
        average_ndgc += ndgc
        average_precision += precision
    
    mAP /= count
    gmap = gmap**(1.0/count)
    average_ndgc/= count
    average_precision /= count


    print 'mAP : ', mAP
    print 'gmAP : ', gmap
    print 'average NDGC : ', average_ndgc
    print 'average p5 : {}, p10 : {}, p20 : {}'.format(average_precision[0],average_precision[1],average_precision[2])
    return mAP

if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # model = ae_model()
    # print model.query()

    evaluate(100)
