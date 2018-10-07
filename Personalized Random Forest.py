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
path='/Users/lay/Desktop/DHG2016/'
import shutil 
from sklearn.mixture import GMM

gesture_list=['1','3','4','5','6']
num_g=5
subject_list=[str(i) for i in range(1,21)]
position_list=[str(i) for i in range(1,6)]
single_finger_list=['1','3','4','5','6']
path='/Users/lay/Desktop/ICIP/DHG2016/'
length=[1,0.5]
num_joints=22
wrist_index=[1]
palm_index=[2]
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

def fisher_vector(xx, gmm):
    '''
    Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    '''
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covars_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

#training data generation
X=[]
Y=[]
L_dict={}
LL=[]
label=-1
for gesture in gesture_list:
    for subject in subject_list:
        for position in position_list:
            if gesture in single_finger_list:
                finger='1'
            else:
                finger='2'
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
                #A=np.vstack((nl[0:3],nl[3:6],nl[6:9]))
                #R,t=rigid_transform_3D(A)
                #nl=transfer(R,t,nl)
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
                nnnl=nnl
                nnl=np.array(nnl).reshape(9,12)
                nnl=nnl.tolist()
                L.extend(nnl)
            L_dict[gesture+subject+position]=np.array(L)
            LL.extend(L)
K = 128
gmm = GMM(n_components=K, covariance_type='diag')
gmm.fit(np.array(LL))
for gesture in gesture_list:
    label=label+1
    for subject in subject_list:
        for position in position_list:
            fv = fisher_vector(L_dict[gesture+subject+position], gmm)
            X.append(fv.tolist())
            Y.append(label)

#random forest
c=[]
d=[]
train_time=[]
test_time=[]
import time
import heapq 
#hyperparameter to be tuned
for itee in range(1,21):
    optimal=[]
    leng=[]
    start=time.time()
    for nt in range(2,21):
        fg=0
        S={}
        test_subject=[itee]
        ns=test_subject[0]
        for it in range(5):
            import pandas as pd
            fg=fg+1
            dis=pickle.load(open('/Users/lay/Desktop/ICIP/buffer/distance'+str(fg)+'.txt', 'rb'))
            dis=np.array(dis)
            index=[]
            for i in range(1,21):
                index.extend([(i-1)*5])
            new_dis=dis[:,index]
            new_dis=new_dis[index,:]
            dis=new_dis
            dis=dis.tolist()
            ls=heapq.nsmallest(nt,dis[ns-1]) 
            s=[]
            for j in ls:
                s.append(dis[ns-1].index(j)+1)
            S[str(it+1)]=s
        su=set([])
        for i in range(1,6):
            su=su|set(S[str(i)])
        su=su-set([ns])
        train_subject=list(su)
        train_index=[]
        test_index=[]
        for j in range(num_g):
            for i in train_subject:
                train_index.extend(range(100*j+(i-1)*5,100*j+5*i))
        for j in range(num_g):
            for i in test_subject:
                test_index.extend(range(100*j+(i-1)*5,100*j+5*i))
        X_train=[]
        Y_train=[]
        X_test=[]
        Y_test=[]
        cor=0
        for i in train_index:
            X_train.append(X[i])
            Y_train.append(Y[i])
        for j in test_index:
            X_test.append(X[j])
            Y_test.append(Y[j])
        #switch svm and random forest here by importing different classifiers
        #from sklearn import svm
        #clf=svm.LinearSVC()
        #clf.fit(X_train, Y_train) 
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
        clf.fit(X_train, Y_train)
        mid=time.time()
        for i in range(len(X_test)):
            predict=clf.predict([X_test[i]])
            if predict[0]==Y_test[i]:
                cor=cor+1
        #print(str(itee)+'correct rate is'+' '+str(cor/len(X_test)))
        optimal.append(cor/len(X_test))
        leng.append(len(train_subject))
        #c.append(cor/len(X_test))
        end=time.time()
        test_time.append(end-mid)
    eend=time.time()
    train_time.append(eend-start)
    d.append(max(optimal))
    print('accuracy of subject'+str(itee),max(optimal))
    print('training size of subject'+str(itee),leng[optimal.index(max(optimal))])
print('The average accuracy is',np.mean(d))