
#------------------------------------------------------------------
# Advance Data Science Innovation - Mar 2022 
# carol.m.paipa@student.uts.edu.au
# Bestest Beer Predictor - PyTorch, FastAPI and Heroku
#------------------------------------------------------------------
# main api folder
# ~/Projects/adv_dsi/adsi_at2/api

from typing import List, Optional
from fastapi import FastAPI, Query, Response
from fastapi.responses import HTMLResponse
from starlette.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
  
# libraries for PyTorch NN
import torch
from src.models.pytorch import PytorchMultiClass

from enum import Enum
import codecs
import random


# initialise FastAPI()
app = FastAPI()


# read in list of brewery/beer names to convert to number/index
brew_name = pd.read_csv('./data/brewery_name_list.csv')   
beer_style = pd.read_csv('./data/beer_style_list.csv')   
health_tips = pd.read_csv('./data/health_tips.csv')   
health_tips = list(health_tips.squeeze())
 
# convert to list for indexes
brew_index = list(brew_name['0'].squeeze())
     
# load in the saved pytorch model
model_pt = torch.load('../models/pytorch_multi_beer_style.pt')

#------------------------------------------------------------------
# Generate ROOT information page API
#------------------------------------------------------------------

def generate_html_response():
    info_home = codecs.open("./home.html", "r", "utf-8")
    html_content = info_home.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return generate_html_response()


#------------------------------------------------------------------
# Health - random health tips
#------------------------------------------------------------------

@app.get('/health', status_code=200)
def generate_html_response_health():
    rand_id = random.randint(0, 4)
    if rand_id==0:
        html_tip = """<h2>Today's <strong style="background-color: #317399; padding: 0 5px; color: #fff;">random</strong> health tip</h2>"""
    elif rand_id==1:
        html_tip = """<h2><strong style="background-color: #317399; padding: 0 5px; color: #fff;">Today's</strong> random health tip</h2>"""
    elif rand_id==2:
        html_tip = """<h2>Today's random <strong style="background-color: #317399; padding: 0 5px; color: #fff;">health</strong> tip</h2>"""
    elif rand_id==3:
        html_tip = """<h2>Today's random health <strong style="background-color: #317399; padding: 0 5px; color: #fff;">tip</strong></h2>"""
    elif rand_id==4:
        html_tip = """<h2><strong style="background-color: #317399; padding: 0 5px; color: #fff;">Today's random health tip</strong></h2>"""
       
    html_end = """<p style="font-size: 0.8em;"><< refresh page <ctrl-R> for another health tip >></p>"""

    html_tip = f'{html_tip}<p style="font-size: 1.2em;">{health_tips[random.randint(0, len(health_tips))]}</p>{html_end}'
    return HTMLResponse(content=html_tip, status_code=200)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return generate_html_response_health()


#------------------------------------------------------------------
# keep this list of features & target beer_style
# 'brewery_name', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv', 'beer_style'
# examples for testing: 
# - Russian Imperial Stout , Stone Brewing Co.    
#------------------------------------------------------------------

# create a function called format_features() that takes input parameters and will return a dictionary with the names of the features as keys and the inpot parameters as lists
def format_features(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    # convert brewery name into numeric index
    brew_ind = brew_index.index(brewery_name)
    return {
        'brewery_name': [brew_ind],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    }


#------------------------------------------------------------------
# define a function called predict
#------------------------------------------------------------------
def predict_beer(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    # convert the features into a dict dataset
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    # convert observations to Tensor
    obs_dataset = torch.Tensor(np.array(obs))
    # Make predictions
    with torch.no_grad():
         output = model_pt(obs_dataset) 
    # convert output tensor to numpy - use astype to convert to integer
    output = torch.argmax(output).numpy().astype(int) 
    # return { 'Predicted beer style is =>' : beer_style.squeeze()[output] }  
    return output


#------------------------------------------------------------------
# function to count the number of "rows" in a dict()
#------------------------------------------------------------------
def dict_len(dict):
    no_count = sum([1 if isinstance(dict[x], (str, int))
                 else len(dict[x]) 
                 for x in dict])
    return no_count


#------------------------------------------------------------------
# ‘/beer/type/’ (GET): Returning prediction for a single input only  
#------------------------------------------------------------------
@app.get("/beer/type")
def beer_type(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    result = predict_beer(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    # return prediction & convert to proper description
    return {'Predicted beer style is =>': beer_style.squeeze()[result]}


#------------------------------------------------------------------
# ‘/beers/type/’ (GET): Returning predictions for a multiple inputs 
#------------------------------------------------------------------
@app.get("/beers/type/")
def beers_type(beer_input: Optional[List[str]] = Query(None)):
    query_items = { "beer_input" : beer_input }

    # setup dataframe for beer input & output
    df_beer = pd.DataFrame(columns=['brewery_name', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv', 'brew_ind', 'beer_predict', 'beer_style'])

    # extract features from dict input strings - separate input rows from dict
    # run predict() function on one dict{} beer-ratings

    for i in range(dict_len(query_items)):
        beer_val = [value[i] for value in query_items.values()]
        beer_str = ' '.join(beer_val).split(',')
        
        # create single row dict - convert features 1-5 into float 
        # - plus brew_ind for brewery name index
        dict = {'brewery_name': beer_str[0],  
                'review_aroma': float(beer_str[1]), 
                'review_appearance': float(beer_str[2]), 
                'review_palate': float(beer_str[3]), 
                'review_taste': float(beer_str[4]), 
                'beer_abv': float(beer_str[5]),
                'brew_ind': brew_index.index(beer_str[0]),
        }

        # run predictions based on user input
        result = predict_beer(dict['brewery_name'],dict['review_aroma'],dict['review_appearance'],
                              dict['review_palate'],dict['review_taste'],dict['beer_abv'])

        # print(beer_style.squeeze()[result])
        # add prediction to dict{}
        dict.update({
                "beer_predict": result, 
                "beer_style": beer_style.squeeze()[result]
                })
        
        # add dict{} to dataframe()
        df_beer = df_beer.append(dict, ignore_index = True)
        
    # return beef predictions   
    return df_beer['beer_style'].to_dict()


# eg: Russian Imperial Stout , Stone Brewing Co.    

    
"""
# Testing in localhost:8080

docker build -t gmm-fastapi:latest .

docker run -dit --rm --name adsi_at2_fastapi -p 8080:80 gmm-fastapi:latest
docker run --rm --name adsi_at2_fastapi -p 8080:80 gmm-fastapi:latest

http://localhost:8080
http://localhost:8080/docs
http://localhost:8080/health

http://localhost:8080/beer/type?brewery_name=Crown%20Brewing&review_aroma=6&review_appearance=5&review_palate=4.5&review_taste=3.5&beer_abv=1.5

# 1 beer input
https://pure-shelf-34977.herokuapp.com/beer/type?brewery_name=Crown%20Brewing&review_aroma=6&review_appearance=5&review_palate=4.5&review_taste=3.5&beer_abv=1.5

# Multiple beer inputs
https://pure-shelf-34977.herokuapp.com/beers/type?beer_input=Stone%20Brewing%20Co.%2C1.1%2C2.2%2C3.3%2C4.4%2C5.5&beer_input=Crown%20Brewing%2C1.1%2C1.2%2C1.3%2C1.4%2C1.5

https://betterprogramming.pub/how-to-add-multiple-request-and-response-examples-in-fastapi-ce2117eac7ed

https://www.infoworld.com/article/3629409/get-started-with-fastapi.html

"""    
        
