import numpy as np
from math import log

def auc(result,total_relavent):

    rel_so_far = 0.0
    ans = np.zeros(11)
    for i in range(len(result)):

        if(result[i]==1):
            rel_so_far+=1
            precision = rel_so_far/(i+1)
            recall = rel_so_far/total_relavent
            x = int(recall*100 // 10)
            while(x>=0 and x<11 and ans[x]==0):
                ans[x] = precision
                x = x-1
    

    # print ans
    return np.mean(ans)

def calc_ndgc(result):

    sorted_result = sorted(result,reverse = True)

    idgc = sorted_result[0]

    for i in range(1,len(sorted_result)):

        idgc += sorted_result[i]*1.0/(log(i+1,2))
    
    dgc = result[0]

    for i in range(1,len(result)):

        dgc += result[i]*1.0/(log(i+1,2))
    
    return dgc/idgc

def calc_precision(result):

    rel_so_far = 0.0
    for i in range(len(result)):

        if(i==5):
            p5 = rel_so_far/5
        elif (i==10):
            p10 = rel_so_far/10
        elif (i==20):
            p20 = rel_so_far/20
            break
        
        if(result[i]==1):
            rel_so_far+=1
        
        
    return np.array([p5,p10,p20])

if __name__ == '__main__':

    print auc([0,1,0,0,1,0,0,1],3)

    print ndgc([1,1,0,1])