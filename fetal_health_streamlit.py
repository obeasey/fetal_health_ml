# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# set up title
st.title('Fetal Health Classification: A Machine Learning App')

# display main image
st.image('fetal_health_image.gif')

# load pre-trained models from pickle files 
dt_pickle = open('decision_tree.pickle', 'rb')
dt_ml = pickle.load(dt_pickle)
dt_pickle.close()
rf_pickle = open('random_forest.pickle', 'rb')
rf_ml = pickle.load(rf_pickle)
rf_pickle.close()
ada_pickle = open('ada_boost.pickle', 'rb')
ada_ml = pickle.load(ada_pickle)
ada_pickle.close()
soft_pickle = open('soft_voting.pickle', 'rb')
soft_ml = pickle.load(soft_pickle)
soft_pickle.close()

# load in a default file to use as an example
default_df = pd.read_csv('fetal_health.csv')

# creating sidebar for file upload
st.sidebar.header('Fetal Health Feature Inputs')

file_upload = st.sidebar.file_uploader('Upload File')
st.sidebar.write("Ensure your data stricly follows the format outlined below.")
st.sidebar.write((default_df.drop(columns=['fetal_health'])).head())

# define the model model to be used 
model = st.sidebar.selectbox('Select a Model', options=['Random Forest', 'Decision Tree', 'AdaBoost', 'Soft Voting'])

if model == "Decision Tree":
    clf = dt_ml
elif model == "Random Forest":
    clf = rf_ml
elif model == "AdaBoost":
    clf = ada_ml
else:
    clf = soft_ml


# defining function to change cell colors 
def color_func(val):
    if val == "Normal":
        color = 'green'
    elif val == "Suspect":
        color = 'yellow'
    elif val == "Pathological":
        color = 'red'
    else:
        color = ''
    return f'background-color: {color}'


if file_upload is None:
    st.write("Please upload data to proceed")
else:
    st.write("Data was succesfully uploaded")
    upload_df = pd.read_csv(file_upload)
    upload_df.dropna(inplace=True)
    prediction = clf.predict(upload_df)
    probability = clf.predict_proba(upload_df)
    upload_df['Predicted Class'] = prediction
    upload_df.loc[upload_df['Predicted Class'] == 1, 'Predicted Class'] = "Normal"
    upload_df.loc[upload_df['Predicted Class'] == 2, 'Predicted Class'] = "Suspect"
    upload_df.loc[upload_df['Predicted Class'] == 3, 'Predicted Class'] = "Pathological"
    for i, class_label in enumerate(clf.classes_):
        upload_df['Prediction Probability'] = probability[:, i]
    upload_df = upload_df.style.applymap(color_func)
    st.write(upload_df)

# setting up tabs for visuals 
if model == "Decision Tree":
    feature_image = 'feature_dt.svg'
    confusion_image = 'confusion_dt.svg'
    report_df = pd.read_csv('dt_class_report.csv', index_col=0).transpose()
elif model == "Random Forest":
    feature_image = 'feature_rf.svg'
    confusion_image = 'confusion_rf.svg'
    report_df = pd.read_csv('rf_class_report.csv', index_col=0).transpose()
elif model == "AdaBoost":
    feature_image = 'feature_ada.svg'
    confusion_image = 'confusion_ada.svg'
    report_df = pd.read_csv('ada_class_report.csv', index_col=0).transpose()
else:
    feature_image = 'feature_soft.svg'
    confusion_image = 'confusion_soft.svg'
    report_df = pd.read_csv('soft_class_report.csv', index_col=0).transpose()

if file_upload is None:
    pass
else:
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

    with tab1:
        st.write("Feature Importance")
        st.image(feature_image)
        st.caption("Features used in this prediction are ranked by relative importance.")
    with tab2:
        st.write("Confusion Matrix")
        st.image(confusion_image)
        st.caption("Confusion Matrix of model predictions.")
    with tab3:
        st.write("Classification Report")
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score.")

