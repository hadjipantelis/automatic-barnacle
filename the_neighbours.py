# %%
from pandas.core.algorithms import value_counts
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import geopandas as gpd
from data_utils import loaders, analysers, setters

st.title("Find your nearest neighbours for your English geography:")

generate_from_raw = False

# %%
### This is getting TOPOLOGICAL INFORMATION
p_spatial_lexicon = loaders.fetch_spatial_info()

# %%
### This is getting AGE INFORMATION
p_age_popul = loaders.fetch_age_info()

# %%
### This is getting IMD INFORMATION
p_imd = loaders.fetch_imd_info()

# %%
### This is getting Ethnicity INFORMATION
p_eth = loaders.fetch_ethnicity_info()

# %%
### This is getting NEW CASES INFORMATION
p_new_cases = loaders.fetch_new_cases_info( )

# %%
### This is getting the FULL TABLE TO WORK WITH

p_full = (
    p_imd.merge(p_age_popul, on="LSOA Code")
    .merge(p_spatial_lexicon, left_on="LSOA Code", right_on="LSOA11CD")
    .merge(p_eth, on="LSOA Code")
    .merge(p_new_cases, on="LSOA Code")
)

# %%
### This is where the feature aggregation happens:

option_gra = st.selectbox(
    "Should we work at LAD, UTLA, CCG, STP, MSOA, NHS Trust or CAL granularity:",
    ["LAD", "UTLA", "CCG", "STP", "MSOA", "CAL","NHST"],
)

if option_gra == "LAD":
    grouping_cols = ["LAD21CD", "LAD21NM"]
elif option_gra == "CAL":
    grouping_cols = ["CAL21CD", "CAL21NM"]
elif option_gra == "STP":
    grouping_cols = ["STP21CD", "STP21NM"]
elif option_gra == "CCG":
    grouping_cols = [
        "CCG21CD",
        "CCG21NM",
    ]
elif option_gra == "MSOA":
    grouping_cols = ["MSOA11CD_APPX", "MSOA11NM_APPX"]
elif option_gra == "UTLA":
    grouping_cols = ["UTLA21CD", "UTLA21NM"]
elif option_gra == "NHST":
    grouping_cols = ["nhstrustcd", "nhstrustnm"]
else:
    grouping_cols = ["useless geographical unit"]

option_city = st.selectbox(
    f"What is your {option_gra} name:", np.sort(p_full[grouping_cols[1]].unique())
)

p_full_agg = setters.get_geo_agg_dataset(p_full, grouping_cols)
p_full_agg_norm = p_full_agg.copy()

# %%
### Arranging the specifics of the KNNs

option_nn = st.slider("How many nearest neighbors:", 3, 13, 3, 1)
st.write(f"You selected {option_nn} neighbor {option_gra}s around {option_city}.")

feat_options = st.multiselect(
    "Which attributes groups should we use?",
    [
        "Age Proportions",
        "IMD",
        "Space",
        "Population Sizes",
        "Ethnicity Proportions",
        "New Cases Rate (Daily)",
    ],
)

option_pca = st.selectbox("Should we use PCA in our feature space:", ["No", "Yes"])

cols_for_metric = setters.get_cols_for_metrics(feat_options)
 
if (option_pca == "Yes") and (len(cols_for_metric) < 4):
    st.write(f"With just {len(cols_for_metric)} features PCA is option in ignored")
    option_pca = "No"


# %%
### The KNN is implemented

map_data = analysers.get_knn_analysis_results(
    p_full_agg_norm,
    p_full_agg,
    option_nn = option_nn, 
    option_pca = option_pca, 
    option_city = option_city,
    cols_for_metric = cols_for_metric,
    grouping_cols = grouping_cols
)

# Write out the results
st.text(
    f"Methodology-wise: \n"
    + f"1. We aggregate LSOA-level data to the geographical granularity requested. \n"
    + f"2. We make the feature matrix based based the group(s) of features selected. \n"
    + f"3. We normalise each feature to N(0,1) and optionally reduce the feature space using PCA. \n"
    + f"4. We find the K-nearest neighbours."
)

n_neighbours = len(map_data.geography_name[1:])
noise_warning = "But we are fitting on noise!" if "noise" in cols_for_metric else ""
if n_neighbours > 2:
    other_neighbours = ", ".join(map_data["geography_name"][1:n_neighbours].values)
    st.write(
        f"The {option_nn} neighbours for {map_data.geography_name[0]} {option_gra} "
        + f"are: {other_neighbours} and {map_data.geography_name[n_neighbours]}."
        + f" {noise_warning}"
    )

st.write(f"The features to find the neighbours are {list(cols_for_metric)}.")

# %%
### Making the plots
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=np.median(map_data.lati),
            longitude=np.median(map_data.long),
            zoom=5,
            pitch=5,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position=["long", "lati"],
                get_color=[f"lati =={map_data.lati[0]}? 255: 0", 30, 200, 150],
                # auto_highlight=True,
                get_radius=3000,  # seems not to scale automatically
                # pickable=True
            ),
        ],
    )
)

# %%
st.write("The attributes used to find the nearest neighbours are:")

if "noise" not in cols_for_metric:
    st.write("Raw data")
    st.write(
        p_full_agg.loc[
            p_full_agg[grouping_cols[1]].isin(list(map_data.geography_name[0:])),
            [grouping_cols[1]] + list(cols_for_metric),
        ]
    )

    st.write("Normalised data")
    st.write(
        p_full_agg_norm.loc[
            p_full_agg_norm[grouping_cols[1]].isin(list(map_data.geography_name[0:])),
            [grouping_cols[1]] + list(cols_for_metric) + ["distances"],
        ]
    )
else:
    st.write("We are matching on noise. :) ")