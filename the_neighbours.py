# %%
from pandas.core.algorithms import value_counts
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import geopandas as gpd
from data_utils import loaders

st.title("Find your nearest neighbours:")

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
p_new_cases = loaders.fetch_new_cases_info()

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
    "Should we work at LTLA, UTLA, CCG, STP, MSOA or CAL granularity:",
    ["LAD", "UTLA", "CCG", "STP", "MSOA", "CAL"],
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
    grouping_cols = ["UTLA20CD", "UTLA20NM"]
else:
    grouping_cols = ["useless geographical unit"]

option_city = st.selectbox(
    f"What is your {option_gra} name:", np.sort(p_full[grouping_cols[1]].unique())
)

p_full_agg = (
    p_full.groupby(grouping_cols)
    .aggregate(
        {
            "area": np.sum,
            "lati": np.median,
            "long": np.median,
            "pop_all_ages": np.sum,
            "pop_00_15": np.sum,
            "pop_16_29": np.sum,
            "pop_30_44": np.sum,
            "pop_45_64": np.sum,
            "pop_65_plus": np.sum,
            "approx_mean_age": np.mean,
            "imd_full": np.mean,
            "imd_health": np.mean,
            "imd_employ": np.mean,
            "imd_living": np.mean,
            "white": np.sum,
            "asian": np.sum,
            "mixed": np.sum,
            "black": np.sum,
            "other": np.sum,
            "eth_all": np.sum,
        }
    )
    .reset_index()
)


p_full_agg_cases = (
    p_full[["nc_sprng_2021", "nc_smmr_2021"] + grouping_cols]
    .groupby(grouping_cols)
    .quantile([0.25, 0.50, 0.75])
    .reset_index()
    .pivot(
        index=grouping_cols, columns="level_2", values=["nc_sprng_2021", "nc_smmr_2021"]
    )
)
p_full_agg_cases.columns = [
    "".join(str(a1)) for a1 in p_full_agg_cases.columns.to_flat_index()
]
p_full_agg_cases.rename(
    {
        "('nc_sprng_2021', 0.25)": "nc_sprng_2021_q25",
        "('nc_sprng_2021', 0.5)": "nc_sprng_2021_q50",
        "('nc_sprng_2021', 0.75)": "nc_sprng_2021_q75",
        "('nc_smmr_2021', 0.25)": "nc_smmr_2021_q25",
        "('nc_smmr_2021', 0.5)": "nc_smmr_2021_q50",
        "('nc_smmr_2021', 0.75)": "nc_smmr_2021_q75",
    },
    axis="columns",
    inplace=True,
)

p_full_agg = p_full_agg.merge(p_full_agg_cases, on=grouping_cols)

p_full_agg["nc_sprng_2021_q50_rate"] = (
    100_000 * p_full_agg["nc_sprng_2021_q50"] / p_full_agg["pop_all_ages"]
)
p_full_agg["nc_smmr_2021_q50_rate"] = (
    100_000 * p_full_agg["nc_smmr_2021_q50"] / p_full_agg["pop_all_ages"]
)

p_full_agg["nc_sprng_2021_q25_rate"] = (
    100_000 * p_full_agg["nc_sprng_2021_q25"] / p_full_agg["pop_all_ages"]
)
p_full_agg["nc_smmr_2021_q25_rate"] = (
    100_000 * p_full_agg["nc_smmr_2021_q25"] / p_full_agg["pop_all_ages"]
)

p_full_agg["nc_sprng_2021_q75_rate"] = (
    100_000 * p_full_agg["nc_sprng_2021_q75"] / p_full_agg["pop_all_ages"]
)
p_full_agg["nc_smmr_2021_q75_rate"] = (
    100_000 * p_full_agg["nc_smmr_2021_q75"] / p_full_agg["pop_all_ages"]
)

p_full_agg["nc_sprng_2021_iqr_rate"] = (
    p_full_agg["nc_sprng_2021_q75_rate"] - p_full_agg["nc_sprng_2021_q25_rate"]
)
p_full_agg["nc_smmr_2021_iqr_rate"] = (
    p_full_agg["nc_smmr_2021_q75_rate"] - p_full_agg["nc_smmr_2021_q25_rate"]
)


p_full_agg["pop_density"] = np.round(p_full_agg["pop_all_ages"] / p_full_agg["area"], 1)

p_full_agg["pop_00_15_prop"] = np.round(
    p_full_agg["pop_00_15"] / p_full_agg["pop_all_ages"], 3
)
p_full_agg["pop_16_29_prop"] = np.round(
    p_full_agg["pop_16_29"] / p_full_agg["pop_all_ages"], 3
)
p_full_agg["pop_30_44_prop"] = np.round(
    p_full_agg["pop_30_44"] / p_full_agg["pop_all_ages"], 3
)
p_full_agg["pop_45_64_prop"] = np.round(
    p_full_agg["pop_45_64"] / p_full_agg["pop_all_ages"], 3
)
p_full_agg["pop_65_plus_prop"] = np.round(
    p_full_agg["pop_65_plus"] / p_full_agg["pop_all_ages"], 3
)

p_full_agg["pop_white_prop"] = np.round(p_full_agg["white"] / p_full_agg["eth_all"], 3)
p_full_agg["pop_asian_prop"] = np.round(p_full_agg["asian"] / p_full_agg["eth_all"], 3)
p_full_agg["pop_mixed_prop"] = np.round(p_full_agg["mixed"] / p_full_agg["eth_all"], 3)
p_full_agg["pop_black_prop"] = np.round(p_full_agg["black"] / p_full_agg["eth_all"], 3)
p_full_agg["pop_other_prop"] = np.round(p_full_agg["other"] / p_full_agg["eth_all"], 3)

p_full_agg["noise"] = np.random.normal(0.0, 1.0, p_full_agg.shape[0])
p_full_agg_norm = p_full_agg.copy()

# %%
### Arranging the specifics of the KNNs

# option_nn = st.selectbox('How many nearest neighbors:', np.arange(3, 13))
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

option_age = "No"
option_imd = "No"
option_spa = "No"
option_pop = "No"
option_eth = "No"
option_case_new = "No"

option_pca = st.selectbox("Should we use PCA in our feature space:", ["No", "Yes"])

if "Age Proportions" in feat_options:
    option_age = "Yes"
if "IMD" in feat_options:
    option_imd = "Yes"
if "Space" in feat_options:
    option_spa = "Yes"
if "Population Sizes" in feat_options:
    option_pop = "Yes"
if "Ethnicity Proportions" in feat_options:
    option_eth = "Yes"
if "New Cases Rate (Daily)" in feat_options:
    option_case_new = "Yes"

cols_for_metric = ["noise"]

if option_age == "Yes":
    cols_for_metric = cols_for_metric + [
        "pop_00_15_prop",
        "pop_16_29_prop",
        "pop_30_44_prop",
        "pop_45_64_prop",
        "pop_65_plus_prop",
    ]

if option_imd == "Yes":
    cols_for_metric = cols_for_metric + [
        "imd_full",
        "imd_health",
        "imd_employ",
        "imd_living",
    ]

if option_spa == "Yes":
    cols_for_metric = cols_for_metric + [
        "area",
        "lati",
        "long",
        "pop_density",
    ]

if option_pop == "Yes":
    cols_for_metric = cols_for_metric + [
        "pop_00_15",
        "pop_16_29",
        "pop_30_44",
        "pop_45_64",
        "pop_65_plus",
        "pop_all_ages",
    ]

if option_eth == "Yes":
    cols_for_metric = cols_for_metric + [
        "pop_white_prop",
        "pop_asian_prop",
        "pop_mixed_prop",
        "pop_black_prop",
        "pop_other_prop",
    ]

if option_case_new == "Yes":
    cols_for_metric = cols_for_metric + [
        "nc_sprng_2021_iqr_rate",
        "nc_sprng_2021_q50_rate",
        "nc_smmr_2021_iqr_rate",
        "nc_smmr_2021_q50_rate",
    ]


if (
    (option_spa == "Yes")
    | (option_imd == "Yes")
    | (option_age == "Yes")
    | (option_pop == "Yes")
    | (option_case_new == "Yes")
    | (option_eth == "Yes")
):
    cols_for_metric.remove("noise")


if (option_pca == "Yes") and (len(cols_for_metric) < 4):
    st.write(f"With just {len(cols_for_metric)} features PCA is option in ignored")
    option_pca = "No"


# %%
### The KNN is implemented

# Scale the input data to be N(0,1)
cols_to_norm = p_full_agg_norm.columns[2:]
p_full_agg_norm[cols_to_norm] = StandardScaler().fit_transform(
    p_full_agg_norm[cols_to_norm]
)

# Initialise the NN 
k = option_nn + 1
nbrs = NearestNeighbors(
    n_neighbors=k,
    algorithm="ball_tree",
)

# Check if PCA is relevant
if (len(cols_for_metric) > 6) and ("No" == option_pca):
    st.write(
        f"We have more than half-a-dozen variables,"
        + " maybe we should use a reduced dimension space."
    )

# Pick only the columns we want for comparison
feat_matrix = p_full_agg_norm[cols_for_metric]

# Compute PCA if requested
if option_pca == "Yes":
    pca_obj = PCA(n_components=feat_matrix.shape[1]).fit(feat_matrix)
    k_comp_used = st.slider(
        "How many components to use:", 2, feat_matrix.shape[1], 2, 1
    )
    _z = np.round(100 * np.sum(pca_obj.explained_variance_ratio_[:k_comp_used]), 1)
    st.write(f"Our PCA space retains roughly {_z}% of the variance.")
    # Replace features used with PCA scores
    feat_matrix = pca_obj.transform(feat_matrix)[:, :k_comp_used]

# Compute the nearest neighbours
nbrs.fit(feat_matrix)
_, indices = nbrs.kneighbors(feat_matrix)

# Find the index of the geography in the p_full_agg_norm data.frame
II = p_full_agg_norm[option_city == p_full_agg_norm[grouping_cols[1]]].index[0]

# Add distances
p_full_agg_norm['distances'] = 0.0
p_full_agg_norm.loc[indices[II],'distances'] = _[II]

map_data = p_full_agg.iloc[indices[II]][[grouping_cols[1], "lati", "long"]].reset_index(
    drop=True
)
map_data["geography_name"] = map_data[grouping_cols[1]]

# Write out the results
st.write(
    f"Methodology-wise: \n "
    + f" 1. We make the feature matrix based on the group(s) of features selected. \n"
    + f"2. We normalise each feature to N(0,1) and optionally reduce the feature space using PCA. \n"
    + f"3. We find the K-nearest neighbours."
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
            [grouping_cols[1]] + list(cols_for_metric) + ['distances'],
        ]
    )
else:
    st.write("We are matching on noise. :) ")
# %%
