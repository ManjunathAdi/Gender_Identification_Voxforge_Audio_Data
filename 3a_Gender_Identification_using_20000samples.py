
print("Bhagavantha Mahamrutyunjaya")

import os
#from joblib import Memory
import numpy as np
from os import walk
from sklearn.mixture import GaussianMixture as GMM
#from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import time
import pickle
from sklearn.model_selection import train_test_split
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB


def ext_features(wavpath):
    "Reads through a mono WAV file, converting each frame to the required features. Returns a 2D array."
    if "wav" in wavpath:
        (rate,sig) = wav.read(wavpath)         
        mfcc_feats = mfcc(sig,rate,numcep=12)
    elif "flac" in wavpath:
        (sig,rate) = sf.read(wavpath)   
        mfcc_feats = mfcc(sig,rate,numcep=12)
    
    return mfcc_feats

   
def train_UBM(M, agg_feats):
    keylist = list(agg_feats.keys())
    ubm_feats = agg_feats[keylist[0]]
    for d in keylist[1:]:
        ubm_feats = np.concatenate((ubm_feats, agg_feats[d]))    
    # === TRAIN UBM ===
    ubm = GMM(n_components=M, covariance_type='full')
    ubm.fit(ubm_feats)

    ubm_params = {}
    ubm_params['means'] = ubm.means_
    ubm_params['weights'] = ubm.weights_
    ubm_params['covariances'] = ubm.covariances_
    ubm_params['precisions'] = ubm.precisions_    
    return ubm_params       

def adapt_UBM_to_speakers(M, speaker_feats, ubm_params):
    # === ADAPT UBM, CONSTRUCT TRAINING FEATURES ===
    speaker_features = {}  
    
    for key, feats in speaker_feats.items():
        #print key, feats.shape   
        updated_means = adapt_UBM(M, ubm_params, feats)
        speaker_features[key] = updated_means
    return speaker_features
    
def adapt_UBM(M, ubm_params, data):
    updated_means = np.array(ubm_params['means'], dtype=np.float32)

    for it in range(1): # adaptation loop
        gmm = GMM(n_components=M, weights_init=ubm_params['weights'], means_init=updated_means, \
                  precisions_init=ubm_params['precisions'], covariance_type='full')
        gmm.fit(data)    
        #  ==== Actual adaptation code =====
        new_means = gmm.means_
        new_weights = gmm.weights_
        T = data.shape[0]
        updated_means = adapt_means(ubm_params['means'], ubm_params['covariances'],\
                                    ubm_params['weights'], new_means, new_weights, T).flatten('C')

    return updated_means    
   
def adapt_means(ubm_means, ubm_covars, ubm_weights, new_means, new_weights, T):
    n_i = new_weights*T
    alpha_i = n_i/(n_i+10)
    new_means[np.isnan(new_means)] = 0.0
    return_means = (alpha_i*new_means.T+(1-alpha_i)*ubm_means.T).T
    
    diag_covars = np.diagonal(ubm_covars, axis1=1, axis2=2)
    
    return_means = ( np.sqrt(ubm_weights) * (1/np.sqrt( diag_covars.T ) ) * return_means.T ).T
    
    return return_means
    
start_time1 = time.time()      

filepath = "Documents"
filename = '20ksamples_df_with_10kMale_10kFemale_voxforge.sav'


df = pickle.load(open(os.path.join(filepath, filename), 'rb'))

X_train, X_test, y_train, y_test = train_test_split(df[["FilePath","Gender"]], df["Gender"], test_size=0.5, stratify=df["Gender"], random_state=42)

trainfilepath = "Documents\\Extracted_data_voxforge\\"

#  Train data preparation      start

trainfeatures={}
for i in range(0, X_train.shape[0]):
    trainfeatures[X_train.iloc[i,0]+'__'+X_train.iloc[i,1]] = ext_features(os.path.join(trainfilepath, X_train.iloc[i,0])) 

M=16    # number of gaussians
D=12    # number of cepstral co-efficients
    
#           Train UBM 16 Gaussians  start
'''
start_time = time.time()
print("\nTraining UBM\n")
ubm_params = train_UBM(M, trainfeatures)

print("---Time Taken to run UBM = %s seconds ---" % (time.time() - start_time))


filename = '16G_ubmprms_12ceps_voxforge___.sav'
#pickle.dump(ubm_params, open(filename, 'wb'))

'''
#          Train UBM 16 Gaussians  end     

# load the 16 G ubm_params
filename = '16G_ubmprms_12ceps_voxforge___.sav'

ubm_params = pickle.load(open(filename, 'rb'))

start_time = time.time()
svm_train_features = adapt_UBM_to_speakers(M, trainfeatures, ubm_params) 
print("---Time Taken to run adapt_UBM_to_speakers = %s seconds ---" % (time.time() - start_time))


train_feats = np.zeros((len(svm_train_features), M*D))
train_labels = np.ones(len(svm_train_features), dtype=np.float32)    
    
i,lablist = 0,[]
for key, features_ in svm_train_features.items():
    train_feats[i] = features_.reshape(1, M*D)
    lablist.append(key)
    if key.split('__')[1] == "Male":         
        #train_labels[i]=1
        train_labels[i]=0
        #print '1', key
    elif key.split('__')[1] == "Female":
        #train_labels[i]=0
        train_labels[i]=1
        #print '-1',key
    i=i+1 
    
    
#  Train data preparation      end    



#     Test data preparation      start

start_time = time.time()
        
testpath = "Documents\\Extracted_data_voxforge\\"

print("\nTest data feature extraction and MAP adaptation \n")    
test_features={}
for i in range(0, X_test.shape[0]):
    testfeats = ext_features(os.path.join(testpath, X_test.iloc[i,0])) 
    spkmeans = adapt_UBM(M, ubm_params, testfeats)
    test_features[X_test.iloc[i,0]+'__'+X_test.iloc[i,1]] = spkmeans.reshape(1, M*D)


print("---Time Taken to run adaptation and extraction = %s seconds ---" % (time.time() - start_time))
        
test_feats = np.zeros((len(test_features), M*D))
test_labels = np.ones(len(test_features), dtype=np.float32)    
    
i,tlablist = 0,[]
for key, features_ in test_features.items():
    test_feats[i] = features_.reshape(1, M*D)
    tlablist.append(key)
    if key.split('__')[1] == "Male":         
        #test_labels[i]=1
        test_labels[i]=0
        #print '1', key
    elif key.split('__')[1] == "Female":
        #test_labels[i]=0
        test_labels[i]=1
        #print '-1',key
    i=i+1 


#   Test data preparation        end      
        
#  Classifier Model Building and testing start     
        
print("\n Train data shape = ",train_feats.shape)      
print("\n Test data shape", test_feats.shape)        
 
       
clf1 = RandomForestClassifier( n_estimators = 300, criterion='gini' )
#clf2 = RandomForestClassifier( n_estimators = 200, criterion='entropy' )
#clf3 = ExtraTreesClassifier( n_estimators = 200, criterion='gini' )
clf4 = ExtraTreesClassifier( n_estimators = 300, criterion='entropy' )
clf5 = svm.SVC(C=1,kernel='linear', probability=True)
clf6 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=500)
clf7 = GaussianNB()

start_time = time.time()
clf1.fit(train_feats, train_labels) 
print("\n---Time Taken to train Randomforest Classifier = %s seconds ---" % (time.time() - start_time))

print(" \n\nCLASSIFIER is RandomForest-Gini")
print("training data Fscore =", f1_score(clf1.predict( train_feats ), train_labels))
print("testing data Fscore =", f1_score(clf1.predict( test_feats ), test_labels))
print("training data logloss =", log_loss(train_labels, clf1.predict_proba(train_feats)[:,1]))
print("testing data logloss =", log_loss(test_labels, clf1.predict_proba(test_feats)[:,1]))
print(" Train Accuracy =", accuracy_score(clf1.predict( train_feats ), train_labels))
print(" Test Accuracy =", accuracy_score(clf1.predict( test_feats ), test_labels))

print("\n", classification_report(clf1.predict( test_feats ), test_labels))

start_time = time.time()
clf4.fit(train_feats, train_labels) 
print("\n---Time Taken to train ExtraTree Classifier = %s seconds ---" % (time.time() - start_time))

print(" \n\nCLASSIFIER is ExtraTree-entropy")
print("training data Fscore =", f1_score(clf4.predict( train_feats ), train_labels))
print("testing data Fscore =", f1_score(clf4.predict( test_feats ), test_labels))
print("training data logloss =", log_loss(train_labels, clf4.predict_proba(train_feats)[:,1]))
print("testing data logloss =", log_loss(test_labels, clf4.predict_proba(test_feats)[:,1]))
print(" Train Accuracy =", accuracy_score(clf4.predict( train_feats ), train_labels))
print(" Test Accuracy =", accuracy_score(clf4.predict( test_feats ), test_labels))

print("\n", classification_report(clf4.predict( test_feats ), test_labels))

start_time = time.time()
clf5.fit(train_feats, train_labels) 
print("\n---Time Taken to train Linear SVM Classifier = %s seconds ---" % (time.time() - start_time))

print(" \n\nCLASSIFIER is Linear SVM")
print("training data Fscore =", f1_score(clf5.predict( train_feats ), train_labels))
print("testing data Fscore =", f1_score(clf5.predict( test_feats ), test_labels))
print("training data logloss =", log_loss(train_labels, clf5.predict_proba(train_feats)[:,1]))
print("testing data logloss =", log_loss(test_labels, clf5.predict_proba(test_feats)[:,1]))
print(" Train Accuracy =", accuracy_score(clf5.predict( train_feats ), train_labels))
print(" Test Accuracy =", accuracy_score(clf5.predict( test_feats ), test_labels))

print("\n", classification_report(clf5.predict( test_feats ), test_labels))

start_time = time.time()
clf6.fit(train_feats, train_labels) 
print("\n---Time Taken to train Gradient Boosting Classifier = %s seconds ---" % (time.time() - start_time))

print(" \n\nCLASSIFIER is Gradient Boosting")
print("training data Fscore =", f1_score(clf6.predict( train_feats ), train_labels))
print("testing data Fscore =", f1_score(clf6.predict( test_feats ), test_labels))
print("training data logloss =", log_loss(train_labels, clf6.predict_proba(train_feats)[:,1]))
print("testing data logloss =", log_loss(test_labels, clf6.predict_proba(test_feats)[:,1]))
print(" Train Accuracy =", accuracy_score(clf6.predict( train_feats ), train_labels))
print(" Test Accuracy =", accuracy_score(clf6.predict( test_feats ), test_labels))

print("\n", classification_report(clf6.predict( test_feats ), test_labels))

start_time = time.time()
clf7.fit(train_feats, train_labels) 
print("\n---Time Taken to train Gaussian NB Classifier = %s seconds ---" % (time.time() - start_time))

print(" \n\nCLASSIFIER is Gaussian NB")
print("training data Fscore =", f1_score(clf7.predict( train_feats ), train_labels))
print("testing data Fscore =", f1_score(clf7.predict( test_feats ), test_labels))
print("training data logloss =", log_loss(train_labels, clf7.predict_proba(train_feats)[:,1]))
print("testing data logloss =", log_loss(test_labels, clf7.predict_proba(test_feats)[:,1]))
print(" Train Accuracy =", accuracy_score(clf7.predict( train_feats ), train_labels))
print(" Test Accuracy =", accuracy_score(clf7.predict( test_feats ), test_labels))

print("\n", classification_report(clf7.predict( test_feats ), test_labels))

print("---Time Taken to run = %s seconds ---" % (time.time() - start_time1))

#  Classifier Model Building and testing      End      

### Results - 
'''
---Time Taken to train Randomforest Classifier = 34.42268657684326 seconds ---
 

CLASSIFIER is RandomForest-Gini
training data Fscore = 1.0
testing data Fscore = 0.952255448555
training data logloss = 0.0646966810533
testing data logloss = 0.202769183812
 Train Accuracy = 1.0
 Test Accuracy = 0.9529

              precision    recall  f1-score   support

        0.0       0.97      0.94      0.95      5135
        1.0       0.94      0.97      0.95      4865

avg / total       0.95      0.95      0.95     10000


---Time Taken to train ExtraTree Classifier = 8.284501791000366 seconds ---
 

CLASSIFIER is ExtraTree-entropy
training data Fscore = 1.0
testing data Fscore = 0.9557684468
training data logloss = 9.99200722163e-16
testing data logloss = 0.209643166243
 Train Accuracy = 1.0
 Test Accuracy = 0.9566

              precision    recall  f1-score   support

        0.0       0.98      0.94      0.96      5188
        1.0       0.94      0.97      0.96      4812

avg / total       0.96      0.96      0.96     10000


---Time Taken to train Linear SVM Classifier = 52.23053741455078 seconds ---
 

CLASSIFIER is Linear SVM
training data Fscore = 0.897548588729
testing data Fscore = 0.898630136986
training data logloss = 0.261555803445
testing data logloss = 0.267790815372
 Train Accuracy = 0.8951
 Test Accuracy = 0.8964

              precision    recall  f1-score   support

        0.0       0.87      0.91      0.89      4780
        1.0       0.92      0.88      0.90      5220

avg / total       0.90      0.90      0.90     10000


---Time Taken to train Gradient Boosting Classifier = 86.31919550895691 seconds ---
 

CLASSIFIER is Gradient Boosting
training data Fscore = 0.99899979996
testing data Fscore = 0.959648421894
training data logloss = 0.0362880203668
testing data logloss = 0.107987738597
 Train Accuracy = 0.999
 Test Accuracy = 0.9596

              precision    recall  f1-score   support

        0.0       0.96      0.96      0.96      4988
        1.0       0.96      0.96      0.96      5012

avg / total       0.96      0.96      0.96     10000


---Time Taken to train Gaussian NB Classifier = 0.049631595611572266 seconds ---
 

CLASSIFIER is Gaussian NB
training data Fscore = 0.735364683301
testing data Fscore = 0.732360679263
training data logloss = 4.57028761923
testing data logloss = 4.5721390002
 Train Accuracy = 0.7794
 Test Accuracy = 0.7762

              precision    recall  f1-score   support

        0.0       0.94      0.71      0.81      6638
        1.0       0.61      0.91      0.73      3362

avg / total       0.83      0.78      0.78     10000
'''


