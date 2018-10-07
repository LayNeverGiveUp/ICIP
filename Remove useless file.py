#change the name of the remove_file_name to remove useless files in each folder
import os
path='/Users/lay/Desktop/ICIP/DHG2016/'
gesture_list=['1','3','4','5','6']
subject_list=[str(i) for i in range(1,21)]
position_list=[str(i) for i in range(1,6)]
remove_file_name='binary_feature.txt'
f=0
for gesture in gesture_list:
    for subject in subject_list:
        for position in position_list:
            new_path=path+'gesture_'+gesture+'/finger_1/subject_'+subject+'/essai_'+position+'/'
            try:
                os.remove(path+'gesture_'+gesture+'/finger_1/subject_'+subject+'/essai_'+position+'/'+remove_file_name)
            except:
                f=f+1