#import modules
import os
import numpy as np
from dtw import dtw
from numpy import linalg as la
from numpy.linalg import norm
import warnings
import matplotlib.pylab as plt
import pickle
import pdb
from hmmlearn import hmm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
path='/Users/lay/Desktop/ICIP/DHG2016/'
import shutil 
np.set_printoptions(threshold=np.inf)  
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#using all subjects and all positions
#same setting with traditional HMM
gesture_list=['1','3','4','5','6']
subject_list=[str(i) for i in range(1,21)]
position_list=[str(i) for i in range(1,6)]
single_finger_list=['1','3','4','5','6']
state_list=[3,3,3,3,3]
num_train=19
testgesture_list=['1','3','4','5','6']
feature_file='feature_1.txt'

#clear buffer
shutil.rmtree('/Users/lay/Desktop/ICIP/buffer') 
os.mkdir('/Users/lay/Desktop/ICIP/buffer')

#generate data structure that gestures all grouped in the form of dictionary with gesture+subject+essai as indexes
#the data has two dictionaries, one for list and the other for numpy
gesture_dict={}
distance_gesture_dict={}
for gesture in gesture_list:
    feature=[]
    length=[]
    for subject in subject_list:
        for position in position_list:
            if gesture in single_finger_list:
                finger='1'
            else:
                finger='2'
            new_path=path+'gesture_'+gesture+'/finger_'+finger+'/subject_'+subject+'/essai_'+position+'/'
            f=open(new_path+feature_file)
            iter_f=iter(f)
            L=[]
            for line in iter_f:
                l=line.split()
                l=[float(i) for i in l]
                L.append(l)
            gesture_dict[subject+'_'+gesture+'_'+position]=L    
            distance_gesture_dict[subject+'_'+gesture+'_'+position]=np.array(L)

# compute distance matrix for each gesture for all subjects (Has Been Calculated)
# Note: just compute this matrix once when you first use it. Just load the distance matrix later, because the computation
#time is quite long!!!!! 
fff=0
for gesture in gesture_list:
    fff=fff+1
    dis=[]
    for subject1 in subject_list:
        for position1 in position_list:
            x1=distance_gesture_dict[subject1+'_'+gesture+'_'+position1]
            dis1=[]
            for subject2 in subject_list:
                for position2 in position_list:
                    x2=distance_gesture_dict[subject2+'_'+gesture+'_'+position2]
                    #pdb.set_trace()
                    dist, cost, acc_cost, path=dtw(x1,x2,dist=lambda x, y: norm(x - y, ord=1))
                    dis1.append(dist)
            dis.append(dis1)
            #store distance matrix of each gesture 
            import pickle
            pickle.dump(dis,open('/Users/lay/Desktop/ICIP/buffer/distance'+str(fff)+'.txt','wb'))
print('Finish')