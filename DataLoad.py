import pandas as pd

#For importing feature extracted data(Processed data)
def data_Load():
    
    processed_data_path= r'Speech-Emotion-Recognition-with-Audio/Processed_Data.csv'

    df=pd.read_csv(processed_data_path)
    
    #Removing Null Values
    df=df.fillna(0)

    #Splitting Data as X and Y for training and testing
    X=df.drop(labels='Emotion',axis=1)

    Y=df['Emotion']
    
    return X,Y

