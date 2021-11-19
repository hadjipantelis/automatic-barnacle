from urllib.request import ftpwrapper
import geopandas as gpd
import pandas as pd
import numpy as np


def fetch_spatial_info(generate_from_raw: bool = False):

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
        return p_spatial_lexicon


def fetch_age_info(generate_from_raw: bool = False):

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
        return p_age_popul


def fetch_imd_info(generate_from_raw: bool = False):

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
        return p_imd


def fetch_ethnicity_info(generate_from_raw: bool = False):

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

        p_eth = p_eth[
            ["LSOA Code", "other", "black", "mixed", "asian", "white", "eth_all"]
        ]

        p_eth.to_csv("data/p_eth.csv", index=False)

    else:

        p_eth = pd.read_csv("data/p_eth.csv")
        return p_eth


def fetch_new_cases_info(generate_from_raw: bool = False):

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

        ## Add mapping corrections:
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

        p_spatial_lexicon = fetch_spatial_info()
        p_age_popul = fetch_age_info()

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
        return p_new_cases
