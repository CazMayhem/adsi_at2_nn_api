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
brew_name = pd.read_csv('data/brewery_name_list.csv')   
beer_style = pd.read_csv('data/beer_style_list.csv')   

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
def format_features(brew_index: int, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    # convert brewery name into numeric index
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
@app.get("/predict/beer")
def predict(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    # convert csv dataframe into series & get the index of the brewery_name
    brew_index = list(brew_name['0'].squeeze())
    brew_index = brew_index.index(brewery_name)
    # convert the features into a dict dataset
    features = format_features(brew_index, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    # convert observations to Tensor
    obs_dataset = torch.Tensor(np.array(obs))
    # Make predictions
    with torch.no_grad():
         output = model_pt(obs_dataset) 
    # convert output tensor to numpy - use astype to convert to integer
    output = torch.argmax(output).numpy().astype(int) 
    # final output
    return { 'Predicted beer style is =>' : beer_style.squeeze()[output] }  

    

"""
# Testing in localhost:8080

docker build -t gmm-fastapi:latest .

docker run -dit --rm --name adsi_at2_fastapi -p 8080:80 gmm-fastapi:latest

http://localhost:8080
http://localhost:8080/docs
http://localhost:8080/predict?brewery_name=Sierra%20Nevada%20Brewing%20Co.&review_aroma=5&review_appearance=5&review_palate=5&review_taste=5&beer_abv=5

"""    
        
