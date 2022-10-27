import pandas as pd

#For importing feature extracted data(Processed data)
def data_Load():
    
    processed_data_path= r'C:\Users\saisa\Documents\Project\Processed_Data.csv'

    df=pd.read_csv(processed_data_path)

    df=df.fillna(0)

    X=df.drop(labels='Emotion',axis=1)

    Y=df['Emotion']
    
    return X,Y

