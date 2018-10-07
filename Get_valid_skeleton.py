#get the valid part of the dynamic gesture sequences according to the start and end time offered 
#by informations_troncage_sequences.txt and store it with the name 'valid_skeleton'
#change the path accoding the dataset location for your case

#the output is a txt file named 'valid_skeleton'
import os
import numpy as np
path='/Users/lay/Desktop/ICIP/DHG2016/'
index_file='/Users/lay/Desktop/ICIP/DHG2016/informations_troncage_sequences.txt'
f=open(index_file)
iter_f=iter(f)
for l1 in iter_f:
    l=l1.split()
    gesture=l[0]
    finger=l[1]
    subject=l[2]
    position=l[3]
    start=int(l[4])
    end=int(l[5])
    new_path=path+'gesture_'+gesture+'/finger_'+finger+'/subject_'+subject+'/essai_'+position+'/'
    ff=open(new_path+'skeleton_world.txt')
    iter_ff=iter(ff)
    L=[]
    for l2 in iter_ff:
        ll=l2.split()
        ll=[float(i) for i in ll]
        L.append(ll)
    L=np.array(L)
    nL=L[range(start-1,end),:]
    np.savetxt(path+'gesture_'+gesture+'/finger_'+finger+'/subject_'+subject+'/essai_'+position+'/'+'valid_skeleton.txt',nL)