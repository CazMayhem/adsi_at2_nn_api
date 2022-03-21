from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import numpy as np

# libraries for PyTorch NN
import torch
from src.models.pytorch import PytorchMultiClass

app = FastAPI()

# ===== LOADING THESE CSV'S & PYTORCH MODEL - IS CRASHING THE API DOCKER =======
# ===== But if I comment them all out - the API continues happily        =======

# read in list of brewery/beer names to convert to number/index
df_brew_name = pd.read_csv('data/brewery_name_list.csv')   
# df_beer_style = pd.read_csv('data/processed/beer_style_list.csv')   

# load in the saved pytorch model
model_pt = torch.load('../models/pytorch_multi_beer_style.pt')

# Lab example for reference
# gmm_pipe = load('../models/gmm_pipeline.joblib')
# ==============================================================================

@app.get("/")
def read_root():
    return {"Hello": "World - Whatsssssupppp"}


@app.get('/health', status_code=200)
def healthcheck():
    return {"Health tip for today": "Apple a day keeps the Doctor away"}


# create a function called format_features() with genre, age, income and spending as input parameters that will return a dictionary with the names of the features as keys and the inpot parameters as lists
def format_features(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    # convert brewery name into numeric index
    brew_index = list(pd.Series(df_brew_name['0'].values)).index(brewery_name)
    return {
        'brewery_name': [brew_index],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    }

# keep this list of features & target beer_style
# 'brewery_name', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv', 'beer_style'

# define a function called predict
@app.get("/beer/predict")
def predict(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    features = format_features(brewery_name,review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    # convert observations to Tensor
    obs_dataset = torch.Tensor(np.array(obs))
    # Make predictions
    output = model_pt(obs_dataset)  
    return JSONResponse(output.tolist()) 
    

"""
docker build -t gmm-fastapi:latest .

http://localhost:8080

review_aroma,review_appearance,review_palate,review_taste,beer_abv
http://localhost:8080/beer/predict/params?brewery_name=Caldera+Brewing+Company&review_aroma=1.5&review_appearance=2.5&review_palate=3.5&review_taste=4.5&beer_abv=5.5

"""    
        
