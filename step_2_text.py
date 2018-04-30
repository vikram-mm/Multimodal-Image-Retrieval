from step1_text import get_textual_features
import os
import cPickle
import numpy as np
import pandas as pd
# from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier


def load_labels():

    class_info = pd.read_csv("ImageCLEFmed2009_train_codes.02.csv")
    get_class = {}
    classes = set()
    path = "dataset/ImageCLEFmed2009_train.02/"
    for img_id,img_class in zip(class_info["image_id"],class_info["05_class"]):
        get_class[path+str(img_id)] = img_class
        classes.add(img_class)
    
    return get_class,classes


def one_hot_text():

    textual_data,textual_vocab = get_textual_features()
    print textual_vocab
    get_class,classes = load_labels()
    # print classes
    print len(classes)

    X = np.zeros((len(textual_data),textual_vocab))
    Y = np.zeros((len(textual_data)))

    i=0
    # print textual_data
    for path, words in textual_data.iteritems():

        for j in words:
            X[i][j] = 1

        label = get_class[path]
        if(label=='\\N'):
            Y[i] = 0
        else:
            Y[i] = int(label)

        # print X[i]
        # print Y

        i+=1

    return X,Y

def get_svm():

    # model = GaussianNB()
   
    model_cache_file = os.path.join('cache', 'svm.pkl')

    if os.path.isfile(model_cache_file):

        print('Loading svm from : ' + model_cache_file)
        with open(model_cache_file, 'rb') as f:
            model = cPickle.load(f)
        print 'done'
        return model
    else:
        
        model = SVC(probability=True)
        X,Y = one_hot_text()
        print 'training ...'
        model.fit(X,Y)

        print model.score(X,Y)
        print('Saving trained model to: ' + model_cache_file)
        with open(model_cache_file, 'wb') as f:
            cPickle.dump(model, f)
        print 'Done!'

        return model







if __name__=='__main__':

    # one_hot_text()
    train_mlp()