# Package import
from DataLoad import data_Load

# Data manipulation libraries
import os
import pandas as pd
import numpy as np

# Machine Learning models family: Classifiers
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Test train split and scalar packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Save and Load models
import joblib

# Exclude warnings
import warnings
warnings.filterwarnings('ignore')

# Loading data
X, Y = data_Load()

# Test and train split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1, stratify = Y,shuffle=True)

# standard scalar
scaler=StandardScaler()

# Fitting scalar on test and trained
X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)
#---------------------- ML Models base & tuned versions ---------

#----------------------- SVC Clasifier --------------------------

model1 = SVC()

model1_1 = SVC(kernel= 'poly', random_state=1, C=0.025, gamma='auto')


#----------------------- MLPC Classifier --------------------------

model2 = MLPClassifier()

model2_1 = MLPClassifier(alpha=0.001, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(512,), learning_rate='adaptive', max_iter=500, solver = 'lbfgs', activation='logistic')

#------------------------KNN Classifier ---------------------------

model3 = KNeighborsClassifier()

model3_1 = KNeighborsClassifier(n_neighbors=5, weights = 'distance', algorithm='ball_tree', leaf_size=50, n_jobs=2)

#------------------------Decision Tree Classifier -----------------

model4 = DecisionTreeClassifier()

model4_1 = DecisionTreeClassifier(criterion='log_loss', splitter='random', class_weight='balanced')

#------------------- Logistic Regression ---------------------------

model5 = LogisticRegression()

model5_1 = LogisticRegression(penalty='elasticnet', C=0.05, class_weight='balanced', solver='saga', max_iter=500, l1_ratio=0.1)

# Models fitting 
model1.fit(X_train, y_train)

# Model save path
filename1 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_SVC_model.sav'

# Saving the model
joblib.dump(model1, filename1)

model1_1.fit(X_train, y_train)

filename2 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_SVC_tuned_model.sav'

joblib.dump(model1_1, filename2)

model2.fit(X_train, y_train)

filename3 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_MLPC_model.sav'

joblib.dump(model2, filename3)

model2_1.fit(X_train, y_train)

filename4 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_tuned_MLPC_model.sav'

joblib.dump(model2_1, filename4)

model3.fit(X_train, y_train)

filename5 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_KNN_model.sav'

joblib.dump(model3, filename5)

model3_1.fit(X_train, y_train)

filename6 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_KNN_tuned_model.sav'

joblib.dump(model3_1, filename6)

model4.fit(X_train, y_train)

filename7 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_DTC_model.sav'

joblib.dump(model4, filename7)

model4_1.fit(X_train, y_train)

filename8 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_DTC_tuned_model.sav'

joblib.dump(model4_1, filename8)

model5.fit(X_train, y_train)

filename9 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_LR_model.sav'

joblib.dump(model5, filename9)

model5_1.fit(X_train, y_train)

filename10 = r'Speech-Emotion-Recognition-with-Audio/models/finalized_LR_tuned_model.sav'

joblib.dump(model5_1, filename10)

print('done fitting')

