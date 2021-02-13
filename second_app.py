# %%
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from sklearn.neighbors import NearestNeighbors

st.title('My second app')

# %%
# Data from:
# https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.73.zip
p_world_cities = pd.read_csv("worldcities.csv")
# p_world_cities= p_world_cities[ "primary" ==p_world_cities.capital]
# p_world_cities= p_world_cities.sort_values(
# "population").drop_duplicates(["iso3"]).reset_index(drop=True)

p_world_cities = p_world_cities[p_world_cities.iso3.isin(['GBR'])]
p_world_cities = p_world_cities[p_world_cities.population >
                                100 * 1000].reset_index(drop=True)

# %%
option_city = st.selectbox(
    'What is your town (assuming it has 100K pop.):', np.sort(p_world_cities.city_ascii))
option_nn = st.selectbox('How many nearst neighbors:', np.arange(3, 13))
st.write(f"You selected {option_nn} neighbors arund {option_city}")

k = option_nn
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
nbrs.fit(p_world_cities[["lat", "lng"]])
_, indices = nbrs.kneighbors(p_world_cities[["lat", "lng"]])

II = p_world_cities[option_city == p_world_cities.city_ascii].index[0]
map_data = p_world_cities.iloc[indices[II]][[
    'city_ascii', 'iso3', "lat", "lng"]].reset_index(drop=True)
map_data.columns = ['city_ascii', 'iso3', 'lat', 'lon']

# %%
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=np.median(map_data.lat),
        longitude=np.median(map_data.lon),
        zoom=4,
        pitch=5,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position=["lon", "lat"],
            get_color=[f'lat =={map_data.lat[0]}? 255: 0', 30, 200, 150],
            auto_highlight=True,
            get_radius=1000,  # seems not to scale automatically
            pickable=True
        ),
    ],
))
# %%
