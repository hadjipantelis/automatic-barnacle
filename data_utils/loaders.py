from urllib.request import ftpwrapper
import geopandas as gpd
import pandas as pd
import numpy as np
import hashlib

def fetch_lsoa_nhs_trust_info(generate_from_raw: bool = False):
    if generate_from_raw:
        #  %%
        # Get catalogue of all NHS Trusts used by the API
        new_cases = pd.read_csv("https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsTrust&metric=newAdmissions&format=csv")
        new_cases.drop_duplicates('areaCode', inplace=True)
        nhs_trusts_catalogue = new_cases[["areaCode", "areaName"]]

        # %%
        # Get UK postcodes to LAT-LONG
        # From https://www.freemaptools.com/download/full-uk-postcodes/ukpostcodes.zip
        # in https://www.freemaptools.com/download-uk-postcode-lat-lng.htm (big file)
        pc_latlong = pd.read_csv("/Users/phadjipa/Downloads/ukpostcodes 2.csv")

        # %%
        # Get UK postcodes to NHS Trusts from NHSD (almost all)
        # From https://files.digital.nhs.uk/assets/ods/current/etrust.zip in
        # https://digital.nhs.uk/services/organisation-data-service/file-downloads/other-nhs-organisations
        nhs_trusts_pc = pd.read_csv("/Users/phadjipa/Downloads/etrust/etrust.csv", header=None)
        nhs_trusts_pc = nhs_trusts_pc.iloc[:, [0,1,9] ]
        nhs_trusts_pc.columns= ['area_code','area_name', 'post_code']

        # %%
        # Add functionality to get some weird postcodes.
        aux_post_codes = dict({
            
            # These four are in from the XLSX https://www.england.nhs.uk/wp-content/uploads/2014/11/nhs-non-nhs-ods-codes.xlsx
            "TAD":"BD18 3LA", # BRADFORD DISTRICT CARE TRUST
            "TAE":"M21 9UN",  # MANCHESTER MENTAL HEALTH AND SOCIAL CARE TRUST
            "TAF":"NW1 0PE",  # CAMDEN AND ISLINGTON NHS FOUNDATION TRUST
            "TAH":"S10 3TH",  # SHEFFIELD HEALTH & SOCIAL CARE NHS FOUNDATION TRUST
            "TAJ":"B70 9PL",  # BLACK COUNTRY PARTNERSHIP NHS FOUNDATION TRUST

            # These are the Nightgale Hospital PH had to look up online.
            "NRRK": "B40 1NT", # NHS Nightingale Hospital Birmingham
            "NR1H" : "E16 1XL", #NHS Nightingale Hospital London (phase 1)	
            "N4H3U": "E16 1XL", # NHS Nightingale Hospital London (phase 2)
            "NR0A": "M3 4LP", #NHS Nightingale Hospital North West
            "NRH8": "EX2 7JG", #NHS Nightingale Hospital Exeter
            "NRR8": "HG1 5LA", # NHS Nightingale Hospital Yorkshire and The Humber  
        
        })
        
        def add_a_few_mode_pcs(x):
            if x in aux_post_codes.keys():
                return aux_post_codes[x]
            else:
                return np.NaN

        # %%
        # Merge NHSD trusts&pcs with the API trusts
        nhs_trusts_pc = nhs_trusts_catalogue.merge(nhs_trusts_pc, how="left", left_on="areaCode", right_on="area_code") 
        nhs_trusts_pc['more_pcs'] = nhs_trusts_pc['areaCode'].apply(lambda x: add_a_few_mode_pcs(x))
        nhs_trusts_pc['filled_post_codes']= nhs_trusts_pc['post_code'].fillna(nhs_trusts_pc['more_pcs'])

        # %%
        # Add long-lat
        nhs_trusts_pc_latlong = nhs_trusts_pc[["filled_post_codes",'areaCode','areaName']].merge(
            pc_latlong[['postcode', 'latitude', 'longitude']], 
            how="left", left_on="filled_post_codes", right_on="postcode")

        area_codes_lat_lon = nhs_trusts_pc_latlong[["latitude","longitude", "areaCode",'areaName']]

        # %%
        # Get LSOA catalogue with LAT-LONG
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

        p_lsoa_location = p_lsoa_location[p_lsoa_location.lsoa11cd.str.startswith('E')]  # England only
        p_lsoa_location.drop(columns=['area'], inplace=True)

        # %%
        # Magic Happens here:
        # (In David Attenborough voice: 
        # "That is cross-join, the most voluminous of all joins. Only in rarified occasions it is seen in the wild.")
        lazy_merge = p_lsoa_location.assign(dummy_var=1).merge(area_codes_lat_lon.assign(dummy_var=1),on='dummy_var')
        lazy_merge.drop(columns=['dummy_var'], inplace=True)
        # Calculate distances between all LSOA and all NHS Trusts (yes, not a Eucleadian geometry I know.)
        lazy_merge['appox_dist'] = np.sqrt((lazy_merge['lati'] - lazy_merge['latitude'])**2 + (lazy_merge['long'] - lazy_merge['longitude'])**2)
        # For each LSOA keep its closest NHS Trust
        p_lsoa_nhs_trust_map = lazy_merge.sort_values('appox_dist').drop_duplicates('lsoa11cd')[['lsoa11cd','areaCode','areaName']].reset_index(drop=True)

        p_lsoa_nhs_trust_map.rename(
                {
                    "areaCode":"nhstrustcd",
                    "areaName":"nhstrustnm",
                },
                axis="columns",
                inplace=True,
            )

        p_lsoa_nhs_trust_map.to_csv("data/p_lsoa_nhs_trust_map.csv", index=False)

    p_lsoa_nhs_trust_map = pd.read_csv("data/p_lsoa_nhs_trust_map.csv")
    return p_lsoa_nhs_trust_map


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
            "/Users/phadjipa/Downloads/Lower_Tier_Local_Authority_to_Upper_Tier_Local_Authority_(April_2021)_Lookup_in_England_and_Wales.csv"
        )
        p_spatial_lexicon = p_spatial_lexicon.merge(
            p_ltla_utla_mapping[["LTLA21CD", "UTLA21CD", "UTLA21NM"]],
            left_on="LAD21CD",
            right_on="LTLA21CD",
        )
        p_spatial_lexicon.drop(columns=["LTLA21CD", "CCG21CDH"], inplace=True)
        del p_ltla_utla_mapping

        p_lsoa_nhs_trust_map  = fetch_lsoa_nhs_trust_info()
        p_spatial_lexicon = p_spatial_lexicon.merge(
            p_lsoa_nhs_trust_map, left_on="LSOA11CD", right_on='lsoa11cd'
        )
        p_spatial_lexicon.drop(columns=["lsoa11cd"], inplace=True)

        p_spatial_lexicon.to_csv("data/p_spatial_lexicon.csv", index=False)

    p_spatial_lexicon = pd.read_csv("data/p_spatial_lexicon.csv")

    # Add MSOA name and MSOA-fake code
    p_spatial_lexicon["MSOA11NM_APPX"] = p_spatial_lexicon.LSOA11NM.apply(
        lambda x: x[: (len(x) - 1)]
    )
    p_spatial_lexicon["MSOA11CD_APPX"] = p_spatial_lexicon.MSOA11NM_APPX.apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[:13]
    )

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

    p_eth = pd.read_csv("data/p_eth.csv")
    return p_eth


def fetch_new_cases_info(generate_from_raw: bool = False, england_only: bool = True, aggregate_to_seasons: bool = True):

    if generate_from_raw:

        # Data from:
        p_new_cases_raw = pd.read_csv(
            "https://api.coronavirus.data.gov.uk/v2/data?areaType=LTLA&metric=newCasesBySpecimenDate&format=csv"
        )

        # Select England only
        if england_only:
            p_new_cases = p_new_cases_raw[p_new_cases_raw.areaCode.str.startswith("E")]
        else:
            p_new_cases  = p_new_cases_raw.copy()

        # Helper function to get season
        def get_season(x):
            if (x >= "2021-03-01") and (x < "2021-06-01"):
                return "spring_2021"
            elif (x >= "2021-06-01") and (x < "2021-09-01"):
                return "summer_2021"
            elif (x >= "2021-09-01") and (x < "2021-12-01"):
                return "autumn_2021"
            elif (x >= "2020-12-01") and (x < "2021-02-01"):
                return "winter_2021"
            elif (x >= "2021-12-01") and (x < "2022-02-01"):
                return "winter_2022"
            else:
                return "other_season"

        # Add season as a variable and drop non-interesting season entries
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

        # Helper function to update LAD codes
        def fix_codes(x):
            if x not in old_to_new_code_dict.keys():
                return x
            else:
                return old_to_new_code_dict[x]

        # Fix LAD codes
        p_new_cases["areaCode"] = p_new_cases["areaCode"].apply(lambda x: fix_codes(x))

        # Perform aggregation based on season (if needed)
        if aggregate_to_seasons:
            p_new_cases = p_new_cases.groupby(["areaCode", "season"]).sum().reset_index()
        else:  
            p_new_cases = p_new_cases.groupby(["areaCode", "date"]).sum().reset_index()

        p_spatial_lexicon = fetch_spatial_info()
        p_age_popul = fetch_age_info()

        # add information about which LSOA we examine
        p_new_cases = p_new_cases.merge(
            p_spatial_lexicon[["LSOA11CD", "LAD21CD"]],
            right_on="LAD21CD",
            left_on="areaCode",
        )

        # add information about what is the underlaying LSOA population we examine
        p_new_cases = p_new_cases.merge(
            p_age_popul[["LSOA Code", "pop_all_ages"]],
            left_on="LSOA11CD",
            right_on="LSOA Code",
        )

        # Aggregate the LSOA populations to get the LAD population
        if aggregate_to_seasons:
            lad_pop = (
                p_new_cases[p_new_cases.season == "spring_2021"]
                .groupby("areaCode")["pop_all_ages"]
                .sum()
                .reset_index()
            )
        else:
            lad_pop = (
                p_new_cases[p_new_cases.date == "2021-08-18"]
                .groupby("areaCode")["pop_all_ages"]
                .sum()
                .reset_index()
            )

        # Basic renaming as this is really LAD population now   
        lad_pop.rename(
            {
                "pop_all_ages": "pop_all_ages_lad",
            },
            axis="columns",
            inplace=True,
        )

        # Add LAD population to the main dataset
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
            "LSOA Code", 
            columns="season" if aggregate_to_seasons else "date", 
            values="approx_lsoa_cases"
        ).reset_index()
        
        if aggregate_to_seasons:
            p_new_cases.rename(
                {
                    "spring_2021": "nc_sprng_2021",
                    "summer_2021": "nc_smmr_2021",
                    "autumn_2021": "nc_atmn_2021",
                    "winter_2021": "nc_wntr_2021",
                    "winter_2022": "nc_wntr_2022",
                },
                axis="columns",
                inplace=True,
            )
        if aggregate_to_seasons: 
            p_new_cases.to_csv("data/p_new_cases.csv", index=False)
        else:
            p_new_cases.to_csv("data/p_new_cases_dates.csv", index=False)

        del p_new_cases_raw

    if aggregate_to_seasons:
        p_new_cases = pd.read_csv("data/p_new_cases.csv")
    else: 
        p_new_cases = pd.read_csv("data/p_new_cases_dates.csv")
        
    return p_new_cases


def fetch_rt_info(generate_from_raw: bool = False):
    if generate_from_raw:
        imperial_rt = pd.read_csv("https://imperialcollegelondon.github.io/covid19local/downloads/UK_hotspot_Rt_estimates.csv")
        imperial_rt.rename(columns=dict({'area': 'la_name'}), inplace=True)
        imperial_rt.date = imperial_rt.date.astype(np.datetime64)
        imperial_rt.drop(columns=["CIlow",	"CIup",	"coverage"], inplace=True)
        
        imperial_rt.to_csv("data/imperial_rt.csv", index=False)
        del imperial_rt

    imperial_rt = pd.read_csv("data/imperial_rt.csv")
    return imperial_rt


def fetch_google_mobility_info(generate_from_raw: bool = False, england_only:bool = False):
    if generate_from_raw:
        _df_goog = pd.read_csv("/Users/phadjipa/Downloads/Global_Mobility_Report(1).csv") # Huge file

        # Make this into a date.time
        _df_goog.date = _df_goog.date.astype(np.datetime64)
        # Subset
        df_goog = _df_goog[(_df_goog.country_region_code == "GB") &  (_df_goog.date >= "2021-08-01")]
        # Mapping from google to LAD
        google_to_lad = pd.read_csv("https://raw.githubusercontent.com/datasciencecampus/google-mobility-reports-data/master/geography/google_mobility_lad_lookup_200903.csv")
        google_to_lad = google_to_lad[~google_to_lad.lad19cd.isna()]
        df_goog = df_goog.merge(google_to_lad[['sub_region_1','lad19cd','la_name']], on='sub_region_1',  how='inner')[
            ["date", "retail_and_recreation_percent_change_from_baseline", "transit_stations_percent_change_from_baseline",
             "workplaces_percent_change_from_baseline", "residential_percent_change_from_baseline", 
              'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',"lad19cd","la_name"]]
        
        df_goog.rename(columns=dict({'retail_and_recreation_percent_change_from_baseline': 'retail_mob',
                                     "transit_stations_percent_change_from_baseline": "transit_mob",
                                     "workplaces_percent_change_from_baseline": "workplace_mob",
                                     "residential_percent_change_from_baseline": "residential_mob",
                                     "parks_percent_change_from_baseline": "parks_mob",
                                     "grocery_and_pharmacy_percent_change_from_baseline": "grocery_mob"
        }), inplace=True)
        if england_only:
                df_goog = df_goog[df_goog.lad19cd.str.startswith('E')]

        df_goog.to_csv("data/df_goog.csv", index=False)
        del df_goog

    df_goog = pd.read_csv("data/df_goog.csv")
    return df_goog
