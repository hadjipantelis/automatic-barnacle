# %%
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from fuzzywuzzy import process, fuzz
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.title('My third app')

#%%
# Data from:
# https://prod-hub-indexer.s3.amazonaws.com/files/1d78d47c87df4212b79fe2323aae8e08/0/full/27700/1d78d47c87df4212b79fe2323aae8e08_0_full_27700.csv
p_lads_location = pd.read_csv("data/Local_Authority_Districts_(December_2019)_Boundaries_UK_BFC.csv")
p_lads_location.drop(["st_lengthshape", "bng_n","bng_e","lad19nmw","objectid" ], axis=1, inplace=True)



#%%
# Data from:
# https://prod-hub-indexer.s3.amazonaws.com/files/1d78d47c87df4212b79fe2323aae8e08/0/full/27700/1d78d47c87df4212b79fe2323aae8e08_0_full_27700.csv
p_lads_deprev = pd.read_excel("data/File_10_-_IoD2019_Local_Authority_District_Summaries__lower-tier__.xlsx",
sheet_name=["IMD","Health","Employment"] ,
     engine='openpyxl')
p_lads_imd = p_lads_deprev['IMD'][["Local Authority District code (2019)",
  "IMD - Average score ", "IMD - Proportion of LSOAs in most deprived 10% nationally "]]
p_lads_imd.columns = ["lad_cd_19", "imd_average_score", "imd_prop_lsoa_10perc"]

p_lads_health = p_lads_deprev['Health'][["Local Authority District code (2019)",
  "Health Deprivation and Disability - Average score ", "Health Deprivation and Disability - Proportion of LSOAs in most deprived 10% nationally "]]
p_lads_health.columns = ["lad_cd_19", "health_average_score", "health_prop_lsoa_10perc"]

p_lads_employment = p_lads_deprev['Employment'][["Local Authority District code (2019)",
  "Employment - Average score ", "Employment - Proportion of LSOAs in most deprived 10% nationally "]]
p_lads_employment.columns = ["lad_cd_19", "employment_average_score", "employment_prop_lsoa_10perc"]

#%%
# Data from:
# 
p_lads_popul = pd.read_excel("data/ukmidyearestimates20192020ladcodes-1.xls", sheet_name="MYE2 - Persons", skiprows=3) 
p_lads_popul.columns = p_lads_popul.iloc[0,:]
p_lads_popul = p_lads_popul.iloc[1:,:]
p_lads_popul["pop_00_15"] = p_lads_popul[np.arange(0, 16)].sum(axis=1)
p_lads_popul["pop_16_75"] = p_lads_popul[np.arange(16, 76)].sum(axis=1)
p_lads_popul["pop_76_90"] = p_lads_popul[[ "90+"] + list(np.arange(76, 90))].sum(axis=1)
p_lads_popul["pop_total"] = p_lads_popul["pop_00_15"]+ p_lads_popul["pop_16_75"]+ p_lads_popul["pop_76_90"]

p_lads_popul["pop_00_15_prop"] = p_lads_popul["pop_00_15"] / p_lads_popul["pop_total"] 
p_lads_popul["pop_16_75_prop"] = p_lads_popul["pop_16_75"] / p_lads_popul["pop_total"] 
p_lads_popul["pop_76_90_prop"] = p_lads_popul["pop_76_90"] / p_lads_popul["pop_total"] 

p_lads_popul["age_mean"] =np.dot(p_lads_popul[ list(np.arange(0, 90))], np.arange(1,91)) / p_lads_popul["pop_total"]
#%% 

p_lads= p_lads_location.merge(p_lads_popul[[
    "Code", "Name","age_mean",
    "pop_total", "pop_00_15", "pop_16_75", "pop_76_90", 
    "pop_00_15_prop", "pop_16_75_prop", "pop_76_90_prop"]], right_on="Code",left_on="lad19cd").merge(
        p_lads_imd, right_on="lad_cd_19",left_on="lad19cd"
    ).merge(
        p_lads_health, right_on="lad_cd_19",left_on="lad19cd"
    ).merge(
        p_lads_employment, right_on="lad_cd_19",left_on="lad19cd"
    )

p_lads.drop(["lad_cd_19", "lad_cd_19_y", "lad_cd_19_x", "Code", "Name"], axis=1, inplace=True)

p_lads["pop_density"] = p_lads["pop_total"] / ( p_lads["st_areashape"] / (1000**2) )

p_lads_norm = p_lads.copy()

cols_to_norm = ["pop_total", "pop_density",
 "age_mean",
"pop_00_15", "pop_16_75", "pop_76_90",   
 "pop_00_15_prop", "pop_16_75_prop", "pop_76_90_prop",
 "long","lat", "st_areashape",
"imd_average_score",  
"imd_prop_lsoa_10perc",        
"health_average_score",       
"health_prop_lsoa_10perc", 
"employment_average_score",       
"employment_prop_lsoa_10perc",
]   
p_lads_norm[cols_to_norm] = StandardScaler().fit_transform(p_lads_norm[cols_to_norm])

np.random.seed(420) 
p_lads_norm["pop_total"] *= 0.50
p_lads_norm["long"] *= 0.50
p_lads_norm["lat"] *= 0.50
p_lads_norm["st_areashape"] *= 0.50
p_lads_norm["noise"] = np.random.normal(0.0, 1.0, p_lads_norm.shape[0])
# p_lads_norm["imd_average_score"] *= 2

# %%
option_city = st.selectbox(
    'What is your LTLA (Lower Tier Local Authority):', np.sort(p_lads.lad19nm))
option_nn = st.selectbox('How many nearest neighbors:', np.arange(3, 13))
st.write(f"You selected {option_nn} neighbors around {option_city}.")


option_age = st.selectbox('Should we use age info:', ["No","Yes"])
option_imd = st.selectbox('Should we use deprevation info:', ["No","Yes"])
option_spa = st.selectbox('Should we use spatial:', ["No","Yes"])   

cols_for_metric = ['noise']

if option_age == "Yes":
    cols_for_metric = cols_for_metric + ["pop_total","age_mean", 
    "pop_00_15", "pop_16_75", "pop_76_90",   
    "pop_00_15_prop", "pop_16_75_prop", "pop_76_90_prop",]

if option_imd == "Yes":
    cols_for_metric = cols_for_metric + ["imd_average_score",  "imd_prop_lsoa_10perc",        
    "health_average_score", "health_prop_lsoa_10perc", 
    "employment_average_score", "employment_prop_lsoa_10perc",]

if option_spa == "Yes":
    cols_for_metric = cols_for_metric + [  "long","lat", "st_areashape","pop_density",]

if (option_spa == "Yes") | (option_imd == "Yes") | (option_age == "Yes"):
    cols_for_metric.remove("noise")

k = option_nn + 1
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',)
nbrs.fit(p_lads_norm[cols_for_metric])
_, indices = nbrs.kneighbors(p_lads_norm[cols_for_metric])

II = p_lads_norm[option_city == p_lads_norm.lad19nm].index[0]
map_data = p_lads.iloc[indices[II]][[
    'lad19nm',  "lat", "long"]].reset_index(drop=True) 



st.write(f"The {option_nn} neighbours for {map_data.lad19nm[0]} are {list(map_data.lad19nm[1:])}.")
st.write(f"The features to find the neighbours are {list(cols_for_metric)}.")  

# %%
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=np.median(map_data.lat),
        longitude=np.median(map_data.long),
        zoom=5,
        pitch=5,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position=["long", "lat"],
            get_color=[f'lat =={map_data.lat[0]}? 255: 0', 30, 200, 150],
            # auto_highlight=True,
            get_radius=3000,  # seems not to scale automatically
            # pickable=True
        ),
    ],
))

# %%
