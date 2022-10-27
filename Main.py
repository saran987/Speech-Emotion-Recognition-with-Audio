# Package import
import pandas as pd
import numpy as np
import os
import librosa
from pydub import AudioSegment

# Packages for Neural Network Models
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler

# Package to save and load a model
import joblib

# Loading processed data for fitting
from DataLoad import data_Load

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# Gloabal access
import glob

# Setting X,y Variables, for train and test split
X,Y = data_Load()

# Splitting data train 70% test 30%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify = Y,shuffle=True)

# Intializing standard scalar
scaler=StandardScaler()

# Fitting scalar
X_train=scaler.fit_transform(X_train)

#Features extraction from Librosa package
# Function to extract features
# Audio files features extraction
def zcr(data):
    
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=2048,hop_length=512)
    
    return np.array(np.squeeze(zcr))

def rms(data):

    rms=librosa.feature.rms(data,frame_length=2048,hop_length=512)

    return np.array(np.squeeze(rms))

def tonnetz(data,sr):

    tonnetz=librosa.feature.tonnetz(data,sr=sr)

    return np.array(np.ravel(tonnetz))

def mfcc(data,sr):

    mfcc=librosa.feature.mfcc(data,sr=sr)

    return np.array(np.ravel(mfcc.T))


#function to get features from MFCC, ZCR, RMS
def extract_features(data,sr):

    result=np.array([])
    
    result = np.append(result,mfcc(data,sr))
    
    result = np.append(result,tonnetz(data,sr))
    
    result = np.append(result, rms(data))
    
    result = np.append(result, zcr(data))
 
    return result

# Function to extract features from individual audio path
def get_features(path,duration=2.5, offset=0.6):
    
    data,sr=librosa.load(path,duration=duration,offset=offset)
    
    aud=extract_features(data,sr)
    
    return aud
  
# Converting uploaded audio to set of features
def result():
    
    X = []
    
    main_path = r'C:/Users/saisa/Documents/Project/upload'
    
    for file in os.listdir(main_path):    
        
        # List of files to convert to wav format      
        
        if file.endswith('mp3'):                                                                 
            
            os.system(f"""ffmpeg -i {file} -acodec pcm_u8 -ar 22050 {file[:-4]}.wav""")      
        
        break
    
    # Extracting features from each wav format file 
    for wav in os.listdir(main_path):
        
        path1 = main_path +  "\\\\" +  str(wav)
        
        features = get_features(path1)
        
        features.resize(1,3024)
        
        break

    return features

# Function that predicts the outcome by loading different trained models.
def output_predictions():

    SVC_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_SVC_model.sav')
    
    SVC_tuned_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_SVC_tuned_model.sav')
        
    MLPC_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_MLPC_model.sav')
    
    MLPC_tuned_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_tuned_MLPC_model.sav')

    KNN_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_KNN_model.sav')
    
    KNN_tuned_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_KNN_tuned_model.sav')
      
    DTC_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_DTC_model.sav')
    
    DTC_tuned_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_DTC_tuned_model.sav')

    LR_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_LR_model.sav')
    
    LR_tuned_model = joblib.load(r'C:\Users\saisa\Documents\Project\models\finalized_LR_tuned_model.sav')
          
    CNN_model = tf.keras.models.load_model(r'C:\Users\saisa\Documents\Project\models\finalized_CNN_model.h5')
    
    CNN_tuned_model = tf.keras.models.load_model(r'C:\Users\saisa\Documents\Project\models\finalized_CNN_tuned_model.h5')
        
    X1 = result()
    
    X1 = scaler.transform(X1)
    
    Emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']
        
    a = SVC_model.predict(X1)[0]
    
    b = SVC_tuned_model.predict(X1)[0]
    
    c = MLPC_model.predict(X1)[0]
    
    d = MLPC_tuned_model.predict(X1)[0]
    
    e = KNN_model.predict(X1)[0]
    
    f = KNN_tuned_model.predict(X1)[0]
    
    g = DTC_model.predict(X1)[0]
    
    h = DTC_tuned_model.predict(X1)[0]

    i = LR_model.predict(X1)[0]
    
    j = LR_tuned_model.predict(X1)[0]
    
    k = CNN_model.predict(X1)[0]
    
    l = CNN_tuned_model.predict(X1)[0]
    
    k = np.argmax(k)
    
    l = np.argmax(l)
    
    k = Emotions[k]
    
    l = Emotions[l]
    
    # Sending Results to fronend    
    return (a,b,c,d,e,f,g,h,i,j,k,l)
