
import pandas as pd   
import streamlit as st   
import numpy as np 
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.model_selection import cross_validate
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
import pickle

html_temp = """
	<div style ="background-color:#b9d3ee; padding:13px">
	<h1 style ="color:#8b4513; text-align:center; ">Car Price Prediction </h1>
	</div>
	"""
# this line allows us to display the front end aspects we have defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)

# Title/Text
#st.markdown("Let's learn how much is your car?")
st.markdown("Hello! Let's learn how much is your car?  :face_with_monocle:")

#Add image
img=Image.open("images.jpeg")
st.image(img, caption="CARS", width =300, use_column_width=True)


# To load machine learning model
filename = "model"
model=pickle.load(open(filename, "rb"))


# To take feature inputs
make_model = st.selectbox("Make Model",['Audi A1', 'Audi A2', 'Audi A3', 'Opel Astra', 'Opel Corsa',
       'Opel Insignia', 'Renault Clio', 'Renault Duster',
       'Renault Espace'])
#age = st.sidebar.number_input("Age:",min_value=0, max_value=3)
age = st.slider("Age", min_value=0, max_value=3, value=1, step=1)
#km = st.sidebar.number_input("Km:",min_value=0, max_value=317000)
km = st.slider("km", min_value=0, max_value=317000, value=0, step=1)
#hp_kw = st.sidebar.number_input("hp_kw:",min_value=40, max_value=294)
hp_kw = st.slider("hp_kw", min_value=40, max_value=294, value=0, step=5)
gears = st.selectbox("Gears", ["5", "6", "7", "8"])


# Create a dataframe using feature inputs
sample = {"make_model": make_model,
           "age": age,
           "km": km,
           'hp_kw': hp_kw,
           'gears': gears
          }

df = pd.DataFrame.from_dict([sample])
st.table(df)


# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(df)
if predict :
    st.success(result[0])