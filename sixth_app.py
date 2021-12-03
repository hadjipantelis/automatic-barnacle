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

st.title("My sixth app")

generate_from_raw = False

# %%
### This is getting TOPOLOGICAL INFORMATION
if generate_from_raw:

    # Data from:
    # http://www.ons.gov.uk/ons/guide-method/geography/products/census/spatial/2011/index.html
    gpd_lsoa_location = gpd.read_file(
        "/Users/phadjipa/Downloads/Lower_layer_super_output_areas_(E+W)_2011_Boundaries_(Generalised_Clipped)_V2/"
        + "LSOA_2011_EW_BGC_V2.shp"
    )
    p_lat_lon = gpd_lsoa_location.geometry.centroid.to_crs(epsg=4326)
    p_area = gpd_lsoa_location.geometry.area / (1000 ** 2)
    p_lsoa_location = pd.DataFrame(
        {
            "lsoa11cd": gpd_lsoa_location.LSOA11CD,
            "area": p_area,
            "lati": p_lat_lon.y,
            "long": p_lat_lon.x,
        }
    )
    del p_area, p_lat_lon, gpd_lsoa_location

    # Data from:
    # https://geoportal.statistics.gov.uk/datasets/ons::lsoa-2011-to-clinical-commissioning-group-to-stp-to-cancer-alliances-april-2021-lookup-in-england/about
    p_geog_mapping = pd.read_csv(
        "/Users/phadjipa/Downloads/LSOA_(2011)_to_Clinical_Commissioning_Group_to_STP_to_Cancer_Alliances_(April_2021)_Lookup_in_England.csv"
    )
    p_spatial_lexicon = p_geog_mapping.merge(
        p_lsoa_location, left_on="LSOA11CD", right_on="lsoa11cd"
    )
    p_spatial_lexicon.drop(columns=["FID", "lsoa11cd"], inplace=True)
    del p_geog_mapping, p_lsoa_location

    # Data from:
    # https://geoportal.statistics.gov.uk/datasets/lower-tier-local-authority-to-upper-tier-local-authority-december-2019-lookup-in-england-and-wales/explore
    p_ltla_utla_mapping = pd.read_csv(
        "/Users/phadjipa/Downloads/Lower_Tier_Local_Authority_to_Upper_Tier_Local_Authority__December_2020__Lookup_in_England_and_Wales.csv"
    )
    p_spatial_lexicon = p_spatial_lexicon.merge(
        p_ltla_utla_mapping[["LTLA20CD", "UTLA20CD", "UTLA20NM"]],
        left_on="LAD21CD",
        right_on="LTLA20CD",
    )
    p_spatial_lexicon.drop(columns=["LTLA20CD"], inplace=True)
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
        engine="openpyxl",
        skiprows=4,
    )
    p_age_popul = p_age_popul[p_age_popul["LSOA Code"].str.startswith("E")]
    p_age_popul["approx_mean_age"] = np.round(
        np.matmul(
            p_age_popul[["0-15", "16-29", "30-44", "45-64", "65+"]].values,
            np.array([7.5, 22.5, 37, 54.5, 72]),
        )
        / p_age_popul["All Ages"],
        2,
    )
    p_age_popul = p_age_popul[
        [
            "All Ages",
            "0-15",
            "16-29",
            "30-44",
            "45-64",
            "65+",
            "approx_mean_age",
            "LSOA Code",
        ]
    ]
    p_age_popul.rename(
        {
            "All Ages": "pop_all_ages",
            "0-15": "pop_00_15",
            "16-29": "pop_16_29",
            "30-44": "pop_30_44",
            "45-64": "pop_45_64",
            "65+": "pop_65_plus",
        },
        axis="columns",
        inplace=True,
    )
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
        engine="openpyxl",
        skiprows=0,
    )
    p_imd = p_imd[
        [
            "LSOA code (2011)",
            "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)",
            "Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)",
            "Employment Decile (where 1 is most deprived 10% of LSOAs)",
            "Living Environment Decile (where 1 is most deprived 10% of LSOAs)",
        ]
    ]
    p_imd.rename(
        {
            "LSOA code (2011)": "LSOA Code",
            "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)": "imd_full",
            "Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)": "imd_health",
            "Employment Decile (where 1 is most deprived 10% of LSOAs)": "imd_employ",
            "Living Environment Decile (where 1 is most deprived 10% of LSOAs)": "imd_living",
        },
        axis="columns",
        inplace=True,
    )
    p_imd.to_csv("data/p_imd.csv", index=False)

else:

    p_imd = pd.read_csv("data/p_imd.csv")

# %%
### This is getting Ethnicity INFORMATION
if generate_from_raw:

    # Data from:
    # https://github.com/RobertASmith/DoPE_Public/blob/master/raw_data/LSOA_Ethnicity.csv
    p_eth = pd.read_csv("/Users/phadjipa/Downloads/LSOA_Ethnicity.csv", skiprows=0)
    p_eth = p_eth[
        ["geography code"]
        + [
            my_col
            for my_col in p_eth.columns
            if "Sex: All persons; Age: All categories" in my_col
        ]
    ]

    p_eth["white"] = p_eth[
        [
            my_col
            for my_col in p_eth.columns
            if ("Ethnic Group: White" in my_col) and (": Total" in my_col)
        ]
    ].sum(axis=1)
    p_eth["asian"] = p_eth[
        [
            my_col
            for my_col in p_eth.columns
            if ("Ethnic Group: Asian" in my_col) and (": Total" in my_col)
        ]
    ].sum(axis=1)
    p_eth["mixed"] = p_eth[
        [
            my_col
            for my_col in p_eth.columns
            if ("Ethnic Group: Mixed" in my_col) and (": Total" in my_col)
        ]
    ].sum(axis=1)
    p_eth["black"] = p_eth[
        [
            my_col
            for my_col in p_eth.columns
            if ("Ethnic Group: Black" in my_col) and (": Total" in my_col)
        ]
    ].sum(axis=1)
    p_eth["other"] = p_eth[
        [
            my_col
            for my_col in p_eth.columns
            if ("Ethnic Group: Other" in my_col) and (": Total" in my_col)
        ]
    ].sum(axis=1)

    p_eth.rename(
        {
            "geography code": "LSOA Code",
            "Sex: All persons; Age: All categories: Age; Ethnic Group: All categories: Ethnic group; measures: Value": "eth_all",
        },
        axis="columns",
        inplace=True,
    )

    p_eth = p_eth[["LSOA Code", "other", "black", "mixed", "asian", "white", "eth_all"]]

    p_eth.to_csv("data/p_eth.csv", index=False)

else:

    p_eth = pd.read_csv("data/p_eth.csv")

# %%
### This is getting NEW CASES INFORMATION
if generate_from_raw:

    # Data from:
    p_new_cases_raw = pd.read_csv(
        "https://api.coronavirus.data.gov.uk/v2/data?areaType=LTLA&metric=newCasesBySpecimenDate&format=csv"
    )

    p_new_cases = p_new_cases_raw[p_new_cases_raw.areaCode.str.startswith("E")]

    def get_season(x):
        if (x >= "2021-03-01") and (x < "2021-06-01"):
            return "spring_2021"
        elif (x >= "2021-06-01") and (x < "2021-09-01"):
            return "summer_2021"
        else:
            return "other_season"

    p_new_cases["season"] = p_new_cases["date"].apply(lambda x: get_season(x))
    p_new_cases = p_new_cases[~p_new_cases.season.str.startswith("other")]

    ## Add mapping:
    old_to_new_code_dict = dict(
        {
            # Buckins codes
            "E07000004": "E06000060",
            "E07000005": "E06000060",
            "E07000006": "E06000060",
            "E07000007": "E06000060",
            # North Northamptonshire
            "E07000156": "E06000061",
            "E07000153": "E06000061",
            "E07000152": "E06000061",
            "E07000150": "E06000061",
            # West Northamptonshire
            "E07000155": "E06000062",
            "E07000154": "E06000062",
            "E07000151": "E06000062",
        }
    )

    def fix_codes(x):
        if x not in old_to_new_code_dict.keys():
            return x
        else:
            return old_to_new_code_dict[x]

    p_new_cases["areaCode"] = p_new_cases["areaCode"].apply(lambda x: fix_codes(x))

    p_new_cases = p_new_cases.groupby(["areaCode", "season"]).sum().reset_index()
    p_new_cases = p_new_cases.merge(
        p_spatial_lexicon[["LSOA11CD", "LAD21CD"]],
        right_on="LAD21CD",
        left_on="areaCode",
    )
    p_new_cases = p_new_cases.merge(
        p_age_popul[["LSOA Code", "pop_all_ages"]],
        left_on="LSOA11CD",
        right_on="LSOA Code",
    )

    lad_pop = (
        p_new_cases[p_new_cases.season == "spring_2021"]
        .groupby("areaCode")["pop_all_ages"]
        .sum()
        .reset_index()
    )
    lad_pop.rename(
        {
            "pop_all_ages": "pop_all_ages_lad",
        },
        axis="columns",
        inplace=True,
    )

    p_new_cases = p_new_cases.merge(lad_pop, on="areaCode")

    # Get approximate LSOA cases assuming that cases are distributed in the
    # LSOAs of an LAD proportionally based on the LSOA population.
    p_new_cases["approx_lsoa_cases"] = np.round(
        p_new_cases["newCasesBySpecimenDate"]
        * p_new_cases["pop_all_ages"]
        / p_new_cases["pop_all_ages_lad"],
        2,
    )

    p_new_cases = p_new_cases.pivot(
        "LSOA Code", columns="season", values="approx_lsoa_cases"
    ).reset_index()
    p_new_cases.rename(
        {"spring_2021": "nc_sprng_2021", "summer_2021": "nc_smmr_2021"},
        axis="columns",
        inplace=True,
    )

    p_new_cases.to_csv("data/p_new_cases.csv", index=False)
    del p_new_cases_raw

else:

    p_new_cases = pd.read_csv("data/p_new_cases.csv")


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
    "Should we work at LTLA, UTLA, CCG, STP or CAL granularity:",
    ["LAD", "UTLA", "CCG", "STP", "CAL"],
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
st.write(f"You selected {option_nn} neighbors around {option_city}.")

# option_age = st.selectbox('Should we use age info:', ["No", "Yes"])
# option_imd = st.selectbox('Should we use deprevation info:', ["No", "Yes"])
# option_spa = st.selectbox('Should we use spatial:', ["No", "Yes"])

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

k = option_nn + 1
nbrs = NearestNeighbors(
    n_neighbors=k,
    algorithm="ball_tree",
)

if (len(cols_for_metric) > 6) and ("No" == option_pca):
    st.write(
        f"We have more than half-a-dozen variables,"
        + " maybe we should use a reduced dimension space."
    )

feat_matrix = p_full_agg_norm[cols_for_metric]
if option_pca == "Yes":
    pca_obj = PCA(n_components=feat_matrix.shape[1]).fit(feat_matrix)
    k_comp_used = st.slider(
        "How many components to use:", 2, feat_matrix.shape[1], 2, 1
    )
    _z = np.round(100 * np.sum(pca_obj.explained_variance_ratio_[:k_comp_used]), 1)
    st.write(f"Our PCA space retains roughly {_z}% of the variance.")
    feat_matrix = pca_obj.transform(feat_matrix)[:, :k_comp_used]

nbrs.fit(feat_matrix)
_, indices = nbrs.kneighbors(feat_matrix)

# st.write(feat_matrix)

II = p_full_agg_norm[option_city == p_full_agg_norm[grouping_cols[1]]].index[0]
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
            [grouping_cols[1]] + list(cols_for_metric),
        ]
    )
else:
    st.write("We are matching on noise. :) ")
# %%
