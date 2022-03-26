#------------------------------------------------------
# DVN AT3 RESEARCH & QUESTIONS
# 90014679 Carol Paipa
# MAY 2020
#------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import datetime
import time
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
from scipy import stats
import sys
import warnings

#if not sys.warnoptions:
#    warnings.simplefilter("ignore")

#------------------------------------------------------

st.title('DVN AT3')

# Load data
# streamlit run '/Users/paipac/Documents/0 DVN/AT3/CarolPaipa_AT3A.py'  
# Local data:    /Users/paipac/Documents/0 DVN/0 Data/

DATE_COLUMN = 'last_review'
DATA_URL = ('who_suicide_statisticsv3.csv')
DATA_GDP = ('income_gdp_ppp_inflation_adjusted.csv')

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

@st.cache
def load_data_gdp():
    data = pd.read_csv(DATA_GDP)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

# data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load 10,000 rows of data into the dataframe.
data = load_data()

data_gdp = load_data_gdp()

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

subset = data
subset_gdp = (data_gdp.set_index('country').stack().reset_index(name='gdp').rename(columns={'level_1':'year'}))
subset_gdp = subset_gdp[subset_gdp['year']>='1979']

#----------------------------------------------------
# set up ranking groups for GDP countries
# https://www.w3resource.com/pandas/dataframe/dataframe-rank.php

subset_country = subset_gdp.groupby('country').mean().reset_index()
subset_country['gdp_avg'] = subset_country['gdp']
subset_country = subset_country.drop(columns='gdp', axis=1)

subset_country['rank_gdp_lowest'] = subset_country['gdp_avg'].rank(ascending=True)
subset_country['rank_gdp_highest'] = subset_country['gdp_avg'].rank(ascending=False)

subset_country['group_rank_lowest'] = (subset_country['rank_gdp_lowest'] // 10 * 10)
subset_country['group_rank_hightest'] = (subset_country['rank_gdp_highest'] // 10 * 10)

#subset_country = subset_country.sort_values("gdp_avg", ascending=False).reset_index()

subset_gdp = subset_gdp.join(subset_country.set_index('country'), on='country')

#------------------------------------------------------
# Sidebar widgets

# Filters UI
age_input = st.sidebar.multiselect(
    'Age',
    data.groupby('age').count().reset_index()['age'].tolist())

country_input = st.sidebar.multiselect(
    'Country',
    data.groupby('country').count().reset_index()['country'].tolist())

year_slider = st.sidebar.slider('Which year(s)?', int(data['year'].min()), int(data['year'].max()), (int(data['year'].min()), int(data['year'].median())))

#zoom_slider = st.sidebar.slider('Zoom', 100, 400, 130)

genre = st.sidebar.radio( "Show GDP countries lowest/highest", ('Lowest GDP', 'Highest GDP'))

gdp_sort = ' Bottom ' if genre == 'Lowest GDP' else ' Top '

gdp_rank = st.sidebar.selectbox(
    'Show' + gdp_sort + 'N GDP countries',
    subset_country.groupby('group_rank_lowest').count().reset_index()['group_rank_lowest'].tolist(),
    index=1)

# Checking subsets
# st.write(subset_gdp)
# st.write(genre,gdp_rank)

# Filter data
subset = data

year_min = year_slider[0]
year_max = year_slider[1]

# by age
if len(age_input) > 0:
    subset = subset[subset['age'].isin(age_input)]

# by country
if len(country_input) > 0:
    subset = subset[subset['country'].isin(country_input)]

if len(country_input) > 0:
    subset_gdp = subset_gdp[subset_gdp['country'].isin(country_input)]

# by year - max year for suicide map data
subset = subset[subset['year'] == year_max]
# by year - range of years for GDP data
subset_gdp = subset_gdp[(subset_gdp['year']>=str(year_min)) & (subset_gdp['year']<=str(year_max))]

if gdp_rank > 0 and genre == 'Lowest GDP':
    subset_gdp = subset_gdp[subset_gdp['group_rank_lowest']<=int(gdp_rank)]

if gdp_rank > 0 and genre == 'Highest GDP':
    subset_gdp = subset_gdp[subset_gdp['group_rank_hightest']<=int(gdp_rank)]

subset["population"] = subset["population"] / 100000

# Show table view of data
st.write('WHO Suicide dataset')
st.write(data.head(5))

st.write('World Bank GDP dataset')
st.write(data_gdp.head(5))

#st.write(subset_gdp['year'].min(), subset_gdp['year'].max())

#------------------------------------------------------
# World map[]

from vega_datasets import data

st.header('How many suicides worldwide?')
st.write('Using Altair Geoshape for visualising by country in ',year_max)

# https://nextjournal.com/sdanisch/cartographic-visualization
#  ['albers', 'albersUsa', 'azimuthalEqualArea', 'azimuthalEquidistant', 'conicConformal', 'conicEqualArea', 'conicEquidistant', 'equirectangular', 'gnomonic', 'identity', 'mercator', 'naturalEarth1', 'orthographic', 'stereographic', 'transverseMercator']

world = data.world_110m.url
countries = alt.topo_feature(world, 'countries')

# show user selected year suicides only
source = subset[subset['year']==int(year_max)].dropna()

#st.write(world)
#st.write(countries)

# joining multipl data sources - data.world_110m.url + suicide.csv
# for country (chloropleth map), year, population
# so all 3 will appear in the tooltip encode step
# https://altair-viz.github.io/user_guide/transform/lookup.html

# colour schemes - plasma, redblue, blueorangeView, spectral, 
# https://vega.github.io/vega/docs/schemes/#categorical

scales = alt.selection_interval(bind='scales')

background = alt.Chart(countries).mark_geoshape().encode(
    color=alt.Color('population:Q', scale=alt.Scale(scheme='spectral',reverse=True), title = "Population (000)"),
    tooltip=['year:N','country:N', 'population:Q']
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(source, 'id', ['population','country','year'])
).project(
    type='mercator',   #equirectangular, mercator
).properties(
    width=750,
    height=530
)

st.altair_chart(background )  #+ points)


#------------------------------------------------------
# Multi-Line Tooltip GDP
# https://altair-viz.github.io/gallery/multiline_tooltip.html
from functools import reduce

source = subset_gdp[['year','country','gdp']]
source['country_gdp'] = subset_gdp['country'] + ' ' + subset_gdp['gdp'].astype(str)
#st.write(source)

#source = source.reset_index().melt('year', var_name='country', value_name='gdp')

nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['year'], empty='none')

# The basic line
line = alt.Chart(source).mark_line(interpolate='basis').encode(
    x='year:Q',
    y='gdp:Q',
    color='country:N'
).properties(
    width=780,
    height=500
)

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(source).mark_point().encode(
    x='year:Q',
    opacity=alt.value(0),
).add_selection(
    nearest
)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

# Draw text labels near the points, and highlight based on selection  'gdp:Q'
text = line.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'country_gdp:N', alt.value(' '))
)

# Draw a rule at the location of the selection
rules = alt.Chart(source).mark_rule(color='gray').encode(
    x='year:Q',
).transform_filter(
    nearest
)

# Put the five layers into a chart and bind the data
lineChartMulti = alt.layer(
    line, selectors, points, rules, text
).properties(
    width=600, height=300
)

st.altair_chart(lineChartMulti)