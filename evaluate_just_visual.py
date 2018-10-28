import re
import pandas as pd
import numpy as np
from irma_reader import *
from auc_ap import auc
import os
from lda_image import lda_image


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
    

    data_path = os.path.join('dataset','ImageCLEFmed2009_train.02')
    model = lda_image(data_path)

    mAP = 0.0
    count = 0
    
    for i in range(num_queries):

        result,query = model.query2()
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
            try:
                if(get_class[x[1]]==query_class):
                    simplified_result.append(1)
                else:
                    simplified_result.append(0)
            except:

                continue

        
        AP = auc(simplified_result,class_count[query_class])
        print 'query ', i, ' AP : ', AP
        mAP+=AP
    
    mAP /= count

    print 'mAP : ', mAP
    return mAP
            

if __name__ == '__main__':

    evaluate(1000)