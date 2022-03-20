# import FastAPI from fastapi, JSONResponse from starlette.responses, load from joblib and pandas
from tokenize import Double
from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

# instantiate a FastAPI() class and save it into a variable called app
app = FastAPI()

# load your trained model from models folder and save it into a variable called gmm_pipe
gmm_pipe = load('../models/pytorch_multi_beer_style.lib')

# create a function called read_root() that will return a dictionary with Hello as key and World as value. Add a decorator to it in order to add a GET endpoint to app on the root
@app.get("/")
def read_root():
    return {"Hello": "World - Whatssssuuuuppppp"}


# create a function called healthcheck() that will return GMM Clustering is all ready to go!. Add a decorator to it in order to add a GET endpoint to app on /health with status code 200
@app.get('/health', status_code=200)
def healthcheck():
    return 'GMM Clustering is all ready to go!'


# create a function called format_features() with genre, age, income and spending as input parameters that will return a dictionary with the names of the features as keys and the inpot parameters as lists
def format_features(review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    return {
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    }


@app.get("/beer/predict")
def predict(review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    features = format_features(review_aroma,review_appearance,review_palate,review_taste,beer_abv)
    obs = pd.DataFrame(features)
    #pred = gmm_pipe.predict(obs)
    return 'predict pred(...)'