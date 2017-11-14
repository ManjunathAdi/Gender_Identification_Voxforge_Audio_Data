
import shutil,os
import pandas as pd
import tarfile
from os import walk

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import soundfile as sf
import pickle


path = "Downloads\\voxforge\\"
 
fnames = []
for (dirpath, dirnames, filenames) in walk(path):
    fnames.extend(filenames)        
    
    
  

for f in fnames:
    tar = tarfile.open('Downloads\\voxforge\\'+f)
    for item in tar:
        #print( item)
        tar.extract(item)
       
 
nlist=[]        
for ele in fnames:
    nlist.append(ele[:-4])    


npath = "Documents\\Extracted_data_voxforge\\"
        
allfiles=[]
allfiles = [f for f in sorted(os.listdir(npath))]
         
len(list(set(allfiles) & set(nlist)))





cnt=0
#create a dataframe with filepath and gender
Filename = []
Gender = []

apath = "Documents\\Extracted_data_voxforge\\"
wav_or_not=1
wav_=0
flac=0
for f in allfiles:
    newpath = "Documents\\Extracted_data_voxforge\\"+f+"\\"
    subfiles = [f for f in sorted(os.listdir(newpath))]
    #print(subfiles)
    if "wav" in subfiles:
        wav_=wav_+1
        wav_or_not=1
    elif "flac" in subfiles:
        flac=flac+1
        wav_or_not=0
    
 
    if "wav" in subfiles:
        files = os.listdir("Documents\\Extracted_data_voxforge\\"+f+"\\wav\\")
    elif "flac" in subfiles:
        files = os.listdir("Documents\\Extracted_data_voxforge\\"+f+"\\flac\\")
    
    
    subfiles_in_etc = [fi for fi in sorted(os.listdir("Documents\\Extracted_data_voxforge\\"+f+"\\etc\\"))]
                       
    if "README" in subfiles_in_etc:  
        cnt = cnt + 1                 
        with open("Documents\\Extracted_data_voxforge\\"+f+"\\etc\\README") as gender_file:
            for line in gender_file:
                if(line[0:8] =='Gender: '):
                    gender = line[8:-1]
                    
        
        for i in files:
            if wav_or_not == 1:
                filepath = f+"\\wav\\"+i
                
            else:
                filepath = f+"\\flac\\"+i
               
                
            Filename.append(filepath)
            Gender.append(gender)
            

            
Output = pd.DataFrame({'FilePath':Filename, 'Gender':Gender})              
       
    
Output.shape
#Out[130]: (91718, 2)

Output.Gender.value_counts()
'''
Out[131]: 
Male             58887
male;            11715
[female];         7816
[male];           5372
Female            3824
male              2311
female;            856
female             396
unknown            193
Please Select      189
male;               70
[male];             39
Weiblich            10
adult               10
Male\t              10
make                10
Masculino           10
Name: Gender, dtype: int64
'''

Output = Output.replace("male", "Male")

Output = Output.replace("male;", "Male")
Output = Output.replace("[male];", "Male")
Output = Output.replace("[female];", "Female")
Output = Output.replace("female", "Female")
Output = Output.replace("female;", "Female")

Output = Output.replace("make", "Male")
Output = Output.replace("male; ", "Male")
Output = Output.replace("[male]; ", "Male")
Output = Output.replace("Male\t", "Male")
Output = Output.replace("Masculino", "Male")
Output = Output.replace("Weiblich", "Female")


Output.Gender.value_counts()
'''
Out[141]: 
Male             78424
Female           12902
unknown            193
Please Select      189
adult               10
Name: Gender, dtype: int64
'''


filename = 'Dataframe_voxforge_91718_audio_files_with_gender_and_filepath.sav'
pickle.dump(Output, open(filename, 'wb'))


# load the model from disk
#Voice_df = pickle.load(open(filename, 'rb'))


Maledata=Output[Output["Gender"]=="Male"]

Maledata.shape
#Out[182]: (78424, 2)

Femaledata=Output[Output["Gender"]=="Female"]

Femaledata.shape
#Out[184]: (12902, 2)


########### Selecting only 10,000 rows of Male samples
Male_10k_samples = Maledata.sample(n=10000)
Male_10k_samples.shape
#Out[193]: (10000, 2)

########### Selecting only 10,000 rows of Female samples
Female_10k_samples = Femaledata.sample(n=10000)
Female_10k_samples.shape
#Out[196]: (10000, 2)


########### Concatenating  10,000 rows of Male and Female samples [ Train - test dataset]
data = pd.concat([Male_10k_samples, Female_10k_samples])

data.shape
#Out[200]: (20000, 2)


filename = '20ksamples_df_with_10kMale_10kFemale_voxforge.sav'
pickle.dump(data, open(filename, 'wb'))

        