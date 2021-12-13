import numpy as np
import pandas as pd

def get_cols_for_metrics(feat_options: list):

    option_age = "No"
    option_imd = "No"
    option_spa = "No"
    option_pop = "No"
    option_eth = "No"
    option_case_new = "No"

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
            "nc_atmn_2021_iqr_rate",
            "nc_atmn_2021_q50_rate",
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

    return cols_for_metric

def get_geo_agg_dataset(p_full: pd.DataFrame, grouping_cols : list):
    p_full_agg = (
        p_full.groupby(grouping_cols).aggregate(
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

    seasonal_names = ["nc_sprng_2021", "nc_smmr_2021", "nc_atmn_2021"]
    p_full_agg_cases = (
        p_full[seasonal_names + grouping_cols]
        .groupby(grouping_cols)
        .quantile([0.25, 0.50, 0.75])
        .reset_index()
        .pivot(index=grouping_cols, columns="level_2", values=seasonal_names)
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
            "('nc_atmn_2021', 0.25)": "nc_atmn_2021_q25",
            "('nc_atmn_2021', 0.5)": "nc_atmn_2021_q50",
            "('nc_atmn_2021', 0.75)": "nc_atmn_2021_q75",
        },
        axis="columns",
        inplace=True,
    )

    p_full_agg = p_full_agg.merge(p_full_agg_cases, on=grouping_cols)


    for _name in [
        "nc_sprng_2021_q25",
        "nc_smmr_2021_q25",
        "nc_atmn_2021_q25",
        "nc_sprng_2021_q50",
        "nc_smmr_2021_q50",
        "nc_atmn_2021_q50",
        "nc_sprng_2021_q75",
        "nc_smmr_2021_q75",
        "nc_atmn_2021_q75",
    ]:

        p_full_agg[_name + "_rate"] = (
            100_000 * p_full_agg[_name] / p_full_agg["pop_all_ages"]
        )

    p_full_agg["nc_sprng_2021_iqr_rate"] = (
        p_full_agg["nc_sprng_2021_q75_rate"] - p_full_agg["nc_sprng_2021_q25_rate"]
    )
    p_full_agg["nc_smmr_2021_iqr_rate"] = (
        p_full_agg["nc_smmr_2021_q75_rate"] - p_full_agg["nc_smmr_2021_q25_rate"]
    )
    p_full_agg["nc_atmn_2021_iqr_rate"] = (
        p_full_agg["nc_atmn_2021_q75_rate"] - p_full_agg["nc_atmn_2021_q25_rate"]
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

    return p_full_agg