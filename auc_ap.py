import numpy as np

def auc(result,total_relavent):

    rel_so_far = 0.0
    ans = np.zeros(11)
    for i in range(len(result)):

        if(result[i]==1):
            rel_so_far+=1
            precision = rel_so_far/(i+1)
            recall = rel_so_far/total_relavent
            x = int(recall*100 // 10)
            # print x
            # print precision,' ',recall,' ',x
            while(x>=0 and x<11 and ans[x]==0):
                ans[x] = precision
                x = x-1
    

    # print ans
    return np.mean(ans)

if __name__ == '__main__':

    print auc([0,1,0,0,1,0,0,1],3)

