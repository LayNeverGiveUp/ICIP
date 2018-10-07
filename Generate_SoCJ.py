#generate descriptor and store it
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
warnings.filterwarnings('ignore')
path='/Users/lay/Desktop/ICIP/DHG2016/'
import shutil 

#change the feature file name and generate related descriptors. feature_0 is SoCJ, feature_1 is SoCJ normalised the 
#hand shape, feature_2 is the SoCJ normalised the hand shape and position
#sometimes the feature_1 has a better result than feature_2, so using feature_2 can be risky.
feature_file='feature_1.txt'

gesture_list=['1','3','4','5','6']
subject_list=[str(i) for i in range(1,21)]
position_list=[str(i) for i in range(1,6)]
#for comparison with the CVPR work, both finger_1 and finger_2 should be involved
#single_finger_list=['2','4','5','6']
testgesture_list=['1','3','4','5','6']
num_joints=22

##the setting is based on the dataset description, check the detailed information of the dataset to know the index 
##of each joint
base_index=[3,7,11,15,19]
first_index=[4,8,12,16,20]
second_index=[5,9,13,17,21]
tip_index=[6,10,14,18,22]
tup={}
tup['tup1']=[3,4,5,6]
tup['tup2']=[7,8,9,10]
tup['tup3']=[11,12,13,14]
tup['tup4']=[15,16,17,18]
tup['tup5']=[19,20,21,22]

def handsize_normalization(start,end,length):
    x=start
    y=end
    d=np.linalg.norm(y-x)
    lamda=length/d
    return lamda*(y-x)

def rigid_transform_3D(A):
    import numpy as np
    A=np.mat(A)
    B=np.mat([[0,-1,0],[0,0,0],[0.4,-0.92,0]])
    assert len(A) == len(B)
    N = A.shape[0]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    # special reflection case
    if np.linalg.det(R) < 0:
       #print ("Reflection detected")
        Vt[2,:] *= -1
        R= Vt.T * U.T
    t = -R*centroid_A.T + centroid_B.T
    return R, t

def transfer(R,t,nl):
    import numpy as np
    for i in range(1,23):
        v=nl[(i-1)*3:i*3]
        v=np.mat(v)
        m=R*v.T + np.tile(t, (1, 1))
        nl[(i-1)*3:i*3]=np.array(m.T)
    return nl

#normalize the hand
gesture_dict={}
for gesture in gesture_list:
    for subject in subject_list:
        for position in position_list:
            for finger in ['1','2']:
                new_path=path+'gesture_'+gesture+'/finger_'+finger+'/subject_'+subject+'/essai_'+position+'/'
                f=open(new_path+'valid_skeleton.txt')
                iter_f=iter(f)
                L=[]
                for line in iter_f:
                    l=line.split()
                    l=np.array([float(i) for i in l])
                    nl=np.array([0 for i in range(num_joints*3)],dtype='float')
                    wrist=l[0:3]
                    nl[0:3]=wrist
                    palm=l[3:6]
                    direction=handsize_normalization(wrist,palm,1)
                    nl[3:6]=np.array(nl[0:3])+direction
                    for i in range(5):
                        direction=handsize_normalization(l[3:6],l[(base_index[i]-1)*3:3*base_index[i]],1)
                        nl[(base_index[i]-1)*3:3*base_index[i]]=nl[3:6]+direction
                        direction=handsize_normalization(l[(base_index[i]-1)*3:3*base_index[i]],l[(first_index[i]-1)*3:3*first_index[i]],0.5)
                        nl[(first_index[i]-1)*3:3*first_index[i]]=nl[(base_index[i]-1)*3:3*base_index[i]]+direction
                        direction=handsize_normalization(l[(first_index[i]-1)*3:3*first_index[i]],l[(second_index[i]-1)*3:3*second_index[i]],0.5)
                        nl[(second_index[i]-1)*3:3*second_index[i]]=nl[(first_index[i]-1)*3:3*first_index[i]]+direction
                        direction=handsize_normalization(l[(second_index[i]-1)*3:3*second_index[i]],l[(tip_index[i]-1)*3:3*tip_index[i]],0.5)
                        nl[(tip_index[i]-1)*3:3*tip_index[i]]=nl[(second_index[i]-1)*3:3*second_index[i]]+direction
                    if feature_file=='feature_2.txt':
                        A=np.vstack((nl[0:3],nl[3:6],nl[6:9]))
                        R,t=rigid_transform_3D(A)
                        nl=transfer(R,t,nl)
                    if feature_file=='feature_0.txt':
                        nl=l
                    nnl=[]
                    for ii in range(5):
                        t=tup['tup'+str(ii+1)]
                        buff=nl[3:6]
                        for iii in range(4):
                            ind=t[iii]
                            nnl.append(nl[(ind-1)*3:3*ind]-buff)
                            buff=nl[(ind-1)*3:3*ind]
                    buff=base_index[0]
                    for jj in range(4):
                        ind=base_index[jj+1]
                        nnl.append(nl[(ind-1)*3:3*ind]-buff)
                        buff=nl[(ind-1)*3:3*ind]
                    buff=first_index[0]
                    for jj in range(4):
                        ind=first_index[jj+1]
                        nnl.append(nl[(ind-1)*3:3*ind]-buff)
                        buff=nl[(ind-1)*3:3*ind]
                    buff=second_index[0]
                    for jj in range(4):
                        ind=second_index[jj+1]
                        nnl.append(nl[(ind-1)*3:3*ind]-buff)
                        buff=nl[(ind-1)*3:3*ind]
                    buff=tip_index[0]
                    for jj in range(4):
                        ind=tip_index[jj+1]
                        nnl.append(nl[(ind-1)*3:3*ind]-buff)
                        buff=nl[(ind-1)*3:3*ind]
                    nnl=np.array(nnl).reshape(1,108)
                    nnl=nnl.tolist()
                    L.append(nnl[0])
                L=np.array(L)
                np.savetxt(path+'gesture_'+gesture+'/finger_'+finger+'/subject_'+subject+'/essai_'+position+'/'+feature_file,L)