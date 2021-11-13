# %%
from pandas.core.algorithms import value_counts
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk 
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import geopandas as gpd

st.title('My sixth app')

generate_from_raw = False

# %%
### This is getting TOPOLOGICAL INFORMATION
if generate_from_raw:

    # Data from:
    # http://www.ons.gov.uk/ons/guide-method/geography/products/census/spatial/2011/index.html 
    gpd_lsoa_location = gpd.read_file(
        "/Users/phadjipa/Downloads/Lower_layer_super_output_areas_(E+W)_2011_Boundaries_(Generalised_Clipped)_V2/" +
        "LSOA_2011_EW_BGC_V2.shp") 
    p_lat_lon = gpd_lsoa_location.geometry.centroid.to_crs(epsg=4326)
    p_area = gpd_lsoa_location.geometry.area / (1000**2)
    p_lsoa_location = pd.DataFrame({
        "lsoa11cd": gpd_lsoa_location.LSOA11CD,
        "area": p_area,
        "lati" : p_lat_lon.y,
        "long" : p_lat_lon.x
    }) 
    del p_area, p_lat_lon, gpd_lsoa_location

    # Data from:
    # https://geoportal.statistics.gov.uk/datasets/ons::lsoa-2011-to-clinical-commissioning-group-to-stp-to-cancer-alliances-april-2021-lookup-in-england/about
    p_geog_mapping = pd.read_csv(
        "/Users/phadjipa/Downloads/LSOA_(2011)_to_Clinical_Commissioning_Group_to_STP_to_Cancer_Alliances_(April_2021)_Lookup_in_England.csv") 
    p_spatial_lexicon = p_geog_mapping.merge(p_lsoa_location, left_on="LSOA11CD", right_on="lsoa11cd")
    p_spatial_lexicon.drop(columns=['FID', 'lsoa11cd'], inplace=True)
    del p_geog_mapping, p_lsoa_location

    # Data from:
    # https://geoportal.statistics.gov.uk/datasets/lower-tier-local-authority-to-upper-tier-local-authority-december-2019-lookup-in-england-and-wales/explore
    p_ltla_utla_mapping = pd.read_csv(
        "/Users/phadjipa/Downloads/Lower_Tier_Local_Authority_to_Upper_Tier_Local_Authority__December_2020__Lookup_in_England_and_Wales.csv") 
    p_spatial_lexicon = p_spatial_lexicon.merge(
        p_ltla_utla_mapping[["LTLA20CD", "UTLA20CD","UTLA20NM"]], 
        left_on="LAD21CD", right_on="LTLA20CD")
    p_spatial_lexicon.drop(columns=['LTLA20CD'], inplace=True)
    del p_ltla_utla_mapping

    p_spatial_lexicon.to_csv("data/p_spatial_lexicon.csv", index=False)

else:

    p_spatial_lexicon = pd.read_csv("data/p_spatial_lexicon.csv")

# %%
### This is getting AGE INFORMATION
if generate_from_raw:

    # Data from:
    # https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimatesnationalstatistics
    p_age_popul = pd.read_excel(
        "/Users/phadjipa/Downloads/sape23dt13mid2020lsoabroadagesestimatesunformatted.xlsx",
        sheet_name="Mid-2020 Persons",
        engine='openpyxl',
        skiprows=4)
    p_age_popul = p_age_popul[p_age_popul["LSOA Code"].str.startswith('E')]
    p_age_popul['approx_mean_age'] = np.round(
        np.matmul(p_age_popul[[ '0-15', '16-29', '30-44', '45-64', '65+']].values, 
        np.array([7.5, 22.5, 37, 54.5, 72]) ) / p_age_popul[ 'All Ages'], 2)
    p_age_popul = p_age_popul[["All Ages",'0-15', '16-29', '30-44', '45-64', '65+',	"approx_mean_age","LSOA Code"]]
    p_age_popul.rename({
       "All Ages":"pop_all_ages",
       '0-15':"pop_00_15",
        '16-29':"pop_16_29",
        '30-44':"pop_30_44", 
        '45-64':"pop_45_64",
        '65+':"pop_65_plus",
    }, axis='columns', inplace=True)
    p_age_popul.to_csv("data/p_age_popul.csv", index=False)
else:
    p_age_popul = pd.read_csv("data/p_age_popul.csv")

# %%
### This is getting IMD INFORMATION
if generate_from_raw:

    # Data from:
    # https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019
    p_imd = pd.read_excel(
        "/Users/phadjipa/Downloads/File_2_-_IoD2019_Domains_of_Deprivation.xlsx",
        sheet_name="IoD2019 Domains",
        engine='openpyxl',
        skiprows=0)
    p_imd = p_imd[['LSOA code (2011)', 
       'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)',
       'Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)',
       'Employment Decile (where 1 is most deprived 10% of LSOAs)',
       'Living Environment Decile (where 1 is most deprived 10% of LSOAs)'
       ]]
    p_imd.rename({
        'LSOA code (2011)': 'LSOA Code', 
       'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)': 'imd_full', 
       'Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)': 'imd_health', 
       'Employment Decile (where 1 is most deprived 10% of LSOAs)': 'imd_employ', 
       'Living Environment Decile (where 1 is most deprived 10% of LSOAs)': 'imd_living', 
    }, axis='columns', inplace=True)
    p_imd.to_csv("data/p_imd.csv", index=False)
else:
    p_imd = pd.read_csv("data/p_imd.csv")

# %%
### This is getting the FULL TABLE TO WORK WITH

p_full = p_imd.merge(p_age_popul, on="LSOA Code").merge(
    p_spatial_lexicon, left_on="LSOA Code", right_on="LSOA11CD"
)

# %%
### This is where the magic happens:


option_gra = st.selectbox('Should we work at LTLA, UTLA, CCG, STP or CAL granularity:', ["LAD", "UTLA", "CCG", "STP","CAL"])

if option_gra == "LAD":
    grouping_cols = ["LAD21CD","LAD21NM"]
elif option_gra == "CAL":
    grouping_cols = ["CAL21CD","CAL21NM"]
elif option_gra == "STP":
    grouping_cols = ['STP21CD', 'STP21NM']
elif option_gra == "CCG":
    grouping_cols = ['CCG21CD', 'CCG21NM',]
elif option_gra == "UTLA":
    grouping_cols = ["UTLA20CD","UTLA20NM"]
else:
    grouping_cols = ["useless geographical unit"]

option_city = st.selectbox(
    f'What is your {option_gra} name:', np.sort(p_full[grouping_cols[1]].unique()))

p_full_agg = p_full.groupby(grouping_cols).aggregate({
        "area":np.sum,
        "lati":np.median,
        "long":np.median,
        'pop_all_ages': np.sum, 
        'pop_00_15': np.sum, 
        'pop_16_29': np.sum, 
        'pop_30_44': np.sum, 
        'pop_45_64': np.sum,
        'pop_65_plus': np.sum,
        'approx_mean_age': np.mean,
        'imd_full': np.mean, 
        'imd_health': np.mean, 
        'imd_employ': np.mean, 
        'imd_living': np.mean,
    }).reset_index()

p_full_agg['pop_density'] = np.round(p_full_agg['pop_all_ages']/p_full_agg['area'],1)
p_full_agg["noise"] = np.random.normal(0.0, 1.0, p_full_agg.shape[0]) 
p_full_agg_norm = p_full_agg.copy() 
 

cols_to_norm = p_full_agg_norm.columns[2:]
p_full_agg_norm[cols_to_norm] = StandardScaler().fit_transform(
    p_full_agg_norm[cols_to_norm])

# Arranging the specifics of the KNNs

option_nn = st.selectbox('How many nearest neighbors:', np.arange(3, 13))
st.write(f"You selected {option_nn} neighbors around {option_city}.")

option_age = st.selectbox('Should we use age info:', ["No", "Yes"])
option_imd = st.selectbox('Should we use deprevation info:', ["No", "Yes"])
option_spa = st.selectbox('Should we use spatial:', ["No", "Yes"])
cols_for_metric = ['noise']

if option_age == "Yes":
    cols_for_metric = cols_for_metric + [ "pop_00_15", "pop_16_29", "pop_30_44", 
                                          "pop_45_64", "pop_65_plus", "approx_mean_age"]

if option_imd == "Yes":
    cols_for_metric = cols_for_metric + ["imd_full", "imd_health","imd_employ","imd_living" ]

if option_spa == "Yes":
    cols_for_metric = cols_for_metric + \
        ["area", "lati", "long", "pop_density", ]

if (option_spa == "Yes") | (option_imd == "Yes") | (option_age == "Yes"):
    cols_for_metric.remove("noise")

k = option_nn + 1
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',)


nbrs.fit(p_full_agg_norm[cols_for_metric])
_, indices = nbrs.kneighbors(p_full_agg_norm[cols_for_metric])

II = p_full_agg_norm[option_city == p_full_agg_norm[grouping_cols[1]]].index[0]
map_data = p_full_agg.iloc[indices[II]][[
        grouping_cols[1], "lati", "long"]].reset_index(drop=True)
map_data['geography_name'] = map_data[grouping_cols[1]]
 
 

st.write( f"Methodology-wise: \n "+
         f" 1. We make the feature matrix based on the group(s) of features selected. \n" +
         f"2. We normalise each feature to N(0,1). \n" +
         f"3. We find the K-nearest neighbours.")

st.write(
    f"The {option_nn} neighbours for {map_data.geography_name[0]} {option_gra} are {list(map_data.geography_name[1:])}.")
st.write(f"The features to find the neighbours are {list(cols_for_metric)}.")

# %%
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=np.median(map_data.lati),
        longitude=np.median(map_data.long),
        zoom=5,
        pitch=5,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position=["long", "lati"],
            get_color=[f'lati =={map_data.lati[0]}? 255: 0', 30, 200, 150],
            # auto_highlight=True,
            get_radius=3000,  # seems not to scale automatically
            # pickable=True
        ),
    ],
))

# %%
if 'noise' not in cols_for_metric:
        st.write("Raw data")
        st.write(
            p_full_agg.loc[p_full_agg[grouping_cols[1]].isin(list(map_data.geography_name[0:])),
            [grouping_cols[1]]+list(cols_for_metric)]
        )

        st.write("Normalised data")
        st.write(
            p_full_agg_norm.loc[p_full_agg_norm[grouping_cols[1]].isin(list(map_data.geography_name[0:])),
            [grouping_cols[1]]+list(cols_for_metric)]
        )
else:
    st.write('We are matching on noise. :) ')
# %%
