# Gender_Identification_Voxforge_Audio_Data



STEPS-
1.] Automate the raw data download using web scraping techniques

Script:  audiofilesdownload.py 

Description: 

I copied all the 6244 .tgz file names from the HTML page – http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/    into a csv file “Audiofiles.csv”. 
Now I sweep for every filename of this .csv file and search it in HTML page and click it after finding.
This is how I downloaded all the 6244 .tgz files.

2.] Initial Data cleaning and preparation

Script:  datapreparation_audiofiles.py 

Description:  

-	All the 6244 downloaded .tgz files are read from the source folder/location and unzipped.
-	In the https://github.com/sandvikcode/data-science-take-home page it is said that 62,440 audio .wav files are there. But there are few .tgz files having more than 10 audio files(for example kayray-20070604-wha.tgz has 28 audio files in it) and there are few .flac audio files along with .wav audio files. Therefore, the total number of audio files are 91,718 [audio files are .wav and .flac]
-	A dataframe with filepath(source location of audio file) and gender of each audio file is created.
-	The Gender has some corrections to be done like Male is represented as ‘male’, ‘[male]’, ‘male;’ etc and similarly for female category. After all these corrections, distribution of gender class is Male             78424
Female           12902
unknown            193
Please Select      189
adult               10
-	I have dumped the entire dataframe with 91,718 audio file details as a pickle object to use later
-	I have dumped another dataframe with 20,000 audio file details[10,000 male and 10,000 female] as pickle object.


3.] Feature Extraction, model building and testing 

Script:  Gender_Identification_using_20000samples.py     and   Gender_Identification_using_Fulldata.py

Feature Extraction: I have used MFCC-GMM-MAP Adaptation to extract features from each audio file.
I have worked on speech data in past for speaker verification and identification. People over the globe have been using MFCC(Mel frequency cepstral co-efficients) algorithm to extract the features.
I have used MFCC for feature extraction on many speech datasets and seen the goodness of these features in classification. So I have used MFCC for feature extraction from audio files.
For every audio file we have certain time duration, that is number of frames. For every frame we get cepstral co-efficients.
So for each audio file after applying MFCC we get data as (no. of frames X no. of cepstral coefficients)
For eg, for a file having 482 frames, after applying 12ceps MFCC, we get data/features as (482 * 12)
On Each and every audio file MFCC 2-d features – GMM(Gaussian Mixture Model) is applied.
The number of Gaussians used is 16.
The data[train and test] is MAP adapted using the prior knowledge obtained as 16 Gaussians from UBM.

I have trained UBM(Universal Background Model) on 10000 audio files(5000 Male and 5000 Female).
This UBM is GMM(Gaussian Mixture Model). The details of this UBM will be used by every audio(train and test) file for MAP adaptation and representation. 

Every audio file will get features (1 X no. of Gaussians * no. of ceps)        eg: (1 X 16*12) = (1 X 192)

Model Building :-
I ran 2 architecture designs namely-

1.] With only 20000 audio samples, 10000 train samples and 10000 test samples

Here the train has 5000 male audio samples and 5000 female audio samples.
And test has 5000 male audio samples and 5000 female audio samples.

2.] With complete data

Complete data’s dataframe has 91718 audio samples and the distribution of Gender class is
Male             78424
Female           12902
unknown            193
Please Select      189
adult               10
I consider only male(78424) and female(12902) samples.
So the data has 78424 + 12902 = 91326 samples.
The distribution is skewed and the train-test split is a stratified split to ensure enough proportion female class samples fall in train and test datasets.
I have used 25% data as train data and remaining 75% data as test data. [Reason to use less train data is good results obtained just by using 10000 audio samples in first configuration design.]
The classifiers used to train classification model are Random Forest, Extra-tree, Linear SVM, Gradient Boosting and Gaussian NB.
There are many other classifiers that could be used to classify. I have used selected set of classifiers.


Metrics that I use always for classification task are Fscore and log-loss.

Results-

1.] With only 20000 audio samples, 10000 train samples and 10000 test samples

Classifiers	Random Forest	Extra Tree	Linear SVM	Gradient Boosting	Gaussian NB
Train Fscore	1	1	0.8903	0.9990	0.8108
Test Fscore	0.9538	0.9570	0.8892	0.9594	0.8076
Train Logloss	0.06491	9.992e-16	0.2737	0.03628	4.5888
Test Logloss	0.20267	0.2101	0.2872	0.10826	4.5894
Time taken to train model	33.1402 seconds	8.0701 seconds	52.8355 seconds	85.0526       seconds	0.04361 seconds


2.] With complete data

Classifiers	Random Forest	Extra Tree	Linear SVM	Gradient Boosting	Gaussian NB
Train Fscore	1	1	0.9608	0.9975	0.9411
Test Fscore	0.9782	0.9767	0.9605	0.9848	0.9385
Train Logloss	0.03514	9.992e-16	0.2073	0.02934	2.1902
Test Logloss	0.11719	0.1213	0.2092	0.07088	2.2826
Time taken to train model	115.3908 seconds	20.3226 seconds	186.9234 seconds	224.5452     seconds	0.0907 seconds

