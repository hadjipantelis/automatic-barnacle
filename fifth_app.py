# %%
from pandas.core.algorithms import value_counts
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk 
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.title('My fifth app')



### This is getting TOPOLOGICAL INFORMATION
# %%
# Data from:
# https://prod-hub-indexer.s3.amazonaws.com/files/1d78d47c87df4212b79fe2323aae8e08/0/full/27700/1d78d47c87df4212b79fe2323aae8e08_0_full_27700.csv
p_lads_location = pd.read_csv(
    "data/Local_Authority_Districts_(December_2019)_Boundaries_UK_BFC.csv")
p_lads_location.drop(["st_lengthshape", "bng_n", "bng_e",
                      "lad19nmw", "objectid"], axis=1, inplace=True)


# Data from:
# https://geoportal.statistics.gov.uk/datasets/lower-tier-local-authority-to-upper-tier-local-authority-december-2019-lookup-in-england-and-wales/explore
p_utla_info = pd.read_csv(
    "data/Lower_Tier_Local_Authority_to_Upper_Tier_Local_Authority__December_2019__Lookup_in_England_and_Wales.csv"
)
p_utla_info.rename(str.lower, axis='columns',inplace=True)
p_utla_info.drop(['fid'], axis=1, inplace=True)


print(f"There are {p_utla_info.ltla19cd.nunique()} distinct LAD codes and" +
     f" {p_utla_info.utla19cd.nunique()} distinct UTLA codes in the look-up table.")


p_utla_lad_map = p_lads_location.merge(p_utla_info[["ltla19cd", "utla19cd", "utla19nm"]], 
                                        left_on="lad19cd", right_on="ltla19cd")

p_utla_location = p_utla_lad_map.groupby(['utla19cd','utla19nm']).agg({'long':np.mean, 
                                'lat':np.mean,
                                'st_areashape':np.sum}).reset_index()

print(f"There are {p_lads_location.lad19cd.nunique()} distinct LAD codes in the p_lads_location table and" +
     f" {p_utla_location.utla19cd.nunique()} distinct UTLA codes in the p_utla_location table.")

### This is getting AGE/POPULATION INFORMATION


# %%
# Data from:
# https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland/mid2019april2020localauthoritydistrictcodes/ukmidyearestimates20192020ladcodes.xls
p_lads_popul = pd.read_excel(
    "data/ukmidyearestimates20192020ladcodes-1.xls",
    sheet_name="MYE2 - Persons",
    skiprows=3)
p_lads_popul.columns = p_lads_popul.iloc[0, :]
p_lads_popul = p_lads_popul.iloc[1:427, :]
p_lads_popul = p_lads_popul.loc[p_lads_popul.Code.str.startswith('E'),:]

p_lads_popul['90+'] = p_lads_popul['90+'].astype('float')
p_lads_popul['All ages'] = p_lads_popul['All ages'].astype('float')


p_utla_popul = p_lads_popul.merge(p_utla_info[["ltla19cd", "utla19cd", "utla19nm"]], 
                                        left_on="Code", right_on="ltla19cd")
p_utla_popul.drop(["Code", "Name", "Geography1", "ltla19cd"],axis=1, inplace=True) 
p_utla_popul = p_utla_popul.groupby(['utla19cd','utla19nm']).sum().reset_index()  


def make_pop_columns(p_popul):

    p_popul["pop_00_15"] = p_popul[np.arange(0, 16)].sum(axis=1)
    p_popul["pop_16_75"] = p_popul[np.arange(16, 76)].sum(axis=1)
    p_popul["pop_76_90"] = p_popul[[
        "90+"] + list(np.arange(76, 90))].sum(axis=1)
    p_popul["pop_total"] = p_popul["pop_00_15"] + \
        p_popul["pop_16_75"] + p_popul["pop_76_90"]

    p_popul["pop_00_15_prop"] = p_popul["pop_00_15"] / \
        p_popul["pop_total"]
    p_popul["pop_16_75_prop"] = p_popul["pop_16_75"] / \
        p_popul["pop_total"]
    p_popul["pop_76_90_prop"] = p_popul["pop_76_90"] / \
        p_popul["pop_total"]

    p_popul["age_mean"] = np.dot(p_popul[list(
        np.arange(0, 90))], np.arange(1, 91)) / p_popul["pop_total"]
    
    return p_popul

p_lads_popul = make_pop_columns(p_lads_popul)
p_utla_popul = make_pop_columns(p_utla_popul)

p_utla_info = p_utla_info.merge(p_lads_popul[['Code', 'All ages']], 
                                right_on="Code", left_on="ltla19cd")
 

# p_lads_popul["pop_00_15"] = p_lads_popul[np.arange(0, 16)].sum(axis=1)
# p_lads_popul["pop_16_75"] = p_lads_popul[np.arange(16, 76)].sum(axis=1)
# p_lads_popul["pop_76_90"] = p_lads_popul[[
#     "90+"] + list(np.arange(76, 90))].sum(axis=1)
# p_lads_popul["pop_total"] = p_lads_popul["pop_00_15"] + \
#     p_lads_popul["pop_16_75"] + p_lads_popul["pop_76_90"]

# p_lads_popul["pop_00_15_prop"] = p_lads_popul["pop_00_15"] / \
#     p_lads_popul["pop_total"]
# p_lads_popul["pop_16_75_prop"] = p_lads_popul["pop_16_75"] / \
#     p_lads_popul["pop_total"]
# p_lads_popul["pop_76_90_prop"] = p_lads_popul["pop_76_90"] / \
#     p_lads_popul["pop_total"]

# p_lads_popul["age_mean"] = np.dot(p_lads_popul[list(
#     np.arange(0, 90))], np.arange(1, 91)) / p_lads_popul["pop_total"]



print(f"There are {p_lads_popul.Code.nunique()} distinct Codes in the p_lads_popul table and" +
     f" {p_utla_popul.utla19cd.nunique()} distinct UTLA codes in the p_utla_popul table.")

# %%
# Data from:
# https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/833995/File_10_-_IoD2019_Local_Authority_District_Summaries__lower-tier__.xlsx
p_lads_deprev = pd.read_excel("data/File_10_-_IoD2019_Local_Authority_District_Summaries__lower-tier__.xlsx",
                              sheet_name=["IMD", "Health", "Employment"],
                              engine='openpyxl')
p_lads_imd = p_lads_deprev['IMD'][["Local Authority District code (2019)",
                                   "IMD - Average score ", "IMD - Proportion of LSOAs in most deprived 10% nationally "]]
p_lads_imd.columns = ["lad_cd_19", "imd_average_score", "imd_prop_lsoa_10perc"]

p_lads_health = p_lads_deprev['Health'][["Local Authority District code (2019)",
                                         "Health Deprivation and Disability - Average score ", "Health Deprivation and Disability - Proportion of LSOAs in most deprived 10% nationally "]]
p_lads_health.columns = [
    "lad_cd_19",
    "health_average_score",
    "health_prop_lsoa_10perc"]

p_lads_employment = p_lads_deprev['Employment'][["Local Authority District code (2019)",
                                                 "Employment - Average score ", "Employment - Proportion of LSOAs in most deprived 10% nationally "]]
p_lads_employment.columns = [
    "lad_cd_19",
    "employment_average_score",
    "employment_prop_lsoa_10perc"]

def weighted(x, cols, w="All ages"):
             return pd.Series(np.average(x[cols], weights=x[w], axis=0), cols)

p_utla_imd = p_lads_imd.merge(p_utla_info[['All ages', 'ltla19cd', 'utla19cd']], 
                                            left_on="lad_cd_19", right_on="ltla19cd"
                                           ).groupby('utla19cd').apply(weighted,
                        ['imd_average_score', 'imd_prop_lsoa_10perc']).reset_index()

p_utla_health= p_lads_health.merge(p_utla_info[['All ages', 'ltla19cd', 'utla19cd']], 
                                            left_on="lad_cd_19", right_on="ltla19cd"
                                           ).groupby('utla19cd').apply(weighted,
                        ['health_average_score', 'health_prop_lsoa_10perc']).reset_index()

p_utla_employment = p_lads_employment.merge(p_utla_info[['All ages', 'ltla19cd', 'utla19cd']], 
                                            left_on="lad_cd_19", right_on="ltla19cd"
                                           ).groupby('utla19cd').apply(weighted,
                        ['employment_average_score', 'employment_prop_lsoa_10perc']).reset_index()


p_utla_imd_health_employ = p_utla_imd.merge(p_utla_health, 
                                            on = 'utla19cd').merge(p_utla_employment, 
                                                                   on = 'utla19cd')
p_lads_imd_health_employ = p_lads_imd.merge(p_lads_health, 
                                            on = 'lad_cd_19').merge(p_lads_employment, 
                                                                   on = 'lad_cd_19')

# %%

p_lads = p_lads_location.merge(p_lads_popul[[
    "Code", "Name", "age_mean",
    "pop_total", "pop_00_15", "pop_16_75", "pop_76_90",
    "pop_00_15_prop", "pop_16_75_prop", "pop_76_90_prop"]], right_on="Code", left_on="lad19cd").merge(
        p_lads_imd_health_employ, right_on="lad_cd_19", left_on="lad19cd"
)
p_lads.drop(["lad_cd_19", "Code", "Name"], axis=1, inplace=True)
p_lads["pop_density"] = p_lads["pop_total"] / \
    (p_lads["st_areashape"] / (1000**2))

p_utla = p_utla_location.merge(p_utla_popul[[
    "utla19cd", "age_mean",
    "pop_total", "pop_00_15", "pop_16_75", "pop_76_90",
    "pop_00_15_prop", "pop_16_75_prop", "pop_76_90_prop"]], right_on="utla19cd", left_on="utla19cd").merge(
        p_utla_imd_health_employ, right_on="utla19cd", left_on="utla19cd"
)
# p_utla.drop(["utla19cd", "Code", "Name"], axis=1, inplace=True)
p_utla["pop_density"] = p_lads["pop_total"] / \
    (p_utla["st_areashape"] / (1000**2))


# %%
p_lads_norm = p_lads.copy()
p_utla_norm = p_utla.copy()

cols_to_norm = ["pop_total", "pop_density",
                "age_mean",
                "pop_00_15", "pop_16_75", "pop_76_90",
                "pop_00_15_prop", "pop_16_75_prop", "pop_76_90_prop",
                "long", "lat", "st_areashape",
                "imd_average_score",
                "imd_prop_lsoa_10perc",
                "health_average_score",
                "health_prop_lsoa_10perc",
                "employment_average_score",
                "employment_prop_lsoa_10perc",
                ]
p_lads_norm[cols_to_norm] = StandardScaler(
).fit_transform(p_lads_norm[cols_to_norm])

p_utla_norm[cols_to_norm] = StandardScaler(
).fit_transform(p_utla_norm[cols_to_norm])


np.random.seed(420)
p_lads_norm["noise"] = np.random.normal(0.0, 1.0, p_lads_norm.shape[0])
p_utla_norm["noise"] = np.random.normal(0.0, 1.0, p_utla_norm.shape[0])

p_lads_norm["pop_total"] *= 0.50
p_lads_norm["long"] *= 0.50
p_lads_norm["lat"] *= 0.50
p_lads_norm["st_areashape"] *= 0.50 

p_utla_norm["pop_total"] *= 0.50
p_utla_norm["long"] *= 0.50
p_utla_norm["lat"] *= 0.50
p_utla_norm["st_areashape"] *= 0.50 



# %%


option_gra = st.selectbox('Should we work at LTLA or UTLA granularity:', ["LTLA", "UTLA"])

if option_gra == 'LTLA':
    option_city = st.selectbox(
        'What is your LTLA (Lower Tier Local Authority):', np.sort(p_lads.lad19nm))
else:
    option_city = st.selectbox(
        'What is your UTLA (Upper Tier Local Authority):', np.sort(p_utla.utla19nm))


option_nn = st.selectbox('How many nearest neighbors:', np.arange(3, 13))
st.write(f"You selected {option_nn} neighbors around {option_city}.")

option_age = st.selectbox('Should we use age info:', ["No", "Yes"])
option_imd = st.selectbox('Should we use deprevation info:', ["No", "Yes"])
option_spa = st.selectbox('Should we use spatial:', ["No", "Yes"])

cols_for_metric = ['noise']

if option_age == "Yes":
    cols_for_metric = cols_for_metric + ["pop_total", "age_mean",
                                         #"pop_00_15", "pop_16_75", "pop_76_90",
                                         "pop_00_15_prop", "pop_16_75_prop", "pop_76_90_prop", ]

if option_imd == "Yes":
    cols_for_metric = cols_for_metric + ["imd_average_score", "imd_prop_lsoa_10perc",
                                         "health_average_score", "health_prop_lsoa_10perc",
                                         "employment_average_score", "employment_prop_lsoa_10perc", ]

if option_spa == "Yes":
    cols_for_metric = cols_for_metric + \
        ["long", "lat", "st_areashape", "pop_density", ]

if (option_spa == "Yes") | (option_imd == "Yes") | (option_age == "Yes"):
    cols_for_metric.remove("noise")

k = option_nn + 1
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',)


if option_gra == 'LTLA': 
    nbrs.fit(p_lads_norm[cols_for_metric])
    _, indices = nbrs.kneighbors(p_lads_norm[cols_for_metric])

    II = p_lads_norm[option_city == p_lads_norm.lad19nm].index[0]
    map_data = p_lads.iloc[indices[II]][[
        'lad19nm', "lat", "long"]].reset_index(drop=True)
    map_data['geography_name'] = map_data['lad19nm']
else:
    nbrs.fit(p_utla_norm[cols_for_metric])
    _, indices = nbrs.kneighbors(p_utla_norm[cols_for_metric])

    II = p_utla_norm[option_city == p_utla_norm.utla19nm].index[0]
    map_data = p_utla.iloc[indices[II]][[
        'utla19nm', "lat", "long"]].reset_index(drop=True)
    map_data['geography_name'] = map_data['utla19nm']
 



st.text( f"Methodology-wise: \n 1. We make the feature matrix based on the group(s) of features selected. \n 2. We normalise each feature to N(0,1). \n 3. We find the K-nearest neighbours.")

st.write(
    f"The {option_nn} neighbours for {map_data.geography_name[0]} are {list(map_data.geography_name[1:])}.")
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
if 'noise' not in cols_for_metric:
    if option_gra == 'LTLA': 
        st.write("Raw data")
        st.write(
            p_lads.loc[p_lads["lad19nm"].isin(list(map_data.geography_name[0:])),['lad19nm']+list(cols_for_metric)]
        )

        st.write("Normalised data")
        st.write(
            p_lads_norm.loc[p_lads["lad19nm"].isin(list(map_data.geography_name[0:])),['lad19nm']+list(cols_for_metric)]
        )
    else:
        st.write("Raw data")
        st.write(
            p_utla.loc[p_utla["utla19nm"].isin(list(map_data.geography_name[0:])),['utla19nm']+list(cols_for_metric)]
        )
        st.write("Normalised data")
        st.write(
            p_utla_norm.loc[p_utla["utla19nm"].isin(list(map_data.geography_name[0:])),['utla19nm']+list(cols_for_metric)]
        ) 
else:
    st.write('We are matching on noise. :) ')