# %%
from operator import ge
from seaborn.rcmod import axes_style
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk  
from data_utils import loaders
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide") 

from numpy.random import default_rng
rng = default_rng(123)

st.title("Make a guess for the impact of your restrictions scenario on Rt")
st.write("A Proof-of-Concept")

generate_from_raw = False

st.text(
    f"Methodology-wise: \n"
    + f"1. We aggregate LSOA-level static features to LAD level. \n"
    + f"2. We use Google mobility as time-varying LAD feature and ICL R_t as our time-varying response. \n"
    + f"3. We make the feature matrix X based based the group(s) of features selected and Google mobility. \n"
    + f"4. We fit a small random forest regressor RF to predict R_t based on the available features X. \n"
    + f"5. We use RF to make a prediction about the R_t levels currently(i.e. we just estimate \hat(f)(x)). \n"
    + f"6. We make a new X_new where mobility is changed according our restriction scenario. \n"
    + f"7. We use RF to make a prediction about the R_t levels in our scenario (i.e. we just estimate \hat(f)(x_new)). \n"
    + f"8. We compare our estimates to see if our scenario led to a change in R_t (hopefully).\n"
)

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
### This is getting Google mobility information
p_google_mobility = loaders.fetch_google_mobility_info()

# %%
### This is getting Rt information
p_rt = loaders.fetch_rt_info()
# %%
### This is getting the STATIC feature table to work with

p_full = (
    p_imd.merge(p_age_popul, on="LSOA Code")
    .merge(p_spatial_lexicon, left_on="LSOA Code", right_on="LSOA11CD")
    .merge(p_eth, on="LSOA Code") 
)

# %%
### This is where the STATIC features aggregation happens:

option_gra = "LAD"

if option_gra == "LAD":
    grouping_cols = ["LAD21CD", "LAD21NM"]
else:
    grouping_cols = ["useless geographical unit"]


p_static_agg = (
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

p_static_agg["pop_density"] = np.round(p_static_agg["pop_all_ages"] / p_static_agg["area"], 1)
p_static_agg["pop_00_15_prop"] = np.round(
    p_static_agg["pop_00_15"] / p_static_agg["pop_all_ages"], 3
)
p_static_agg["pop_16_29_prop"] = np.round(
    p_static_agg["pop_16_29"] / p_static_agg["pop_all_ages"], 3
)
p_static_agg["pop_30_44_prop"] = np.round(
    p_static_agg["pop_30_44"] / p_static_agg["pop_all_ages"], 3
)
p_static_agg["pop_45_64_prop"] = np.round(
    p_static_agg["pop_45_64"] / p_static_agg["pop_all_ages"], 3
)
p_static_agg["pop_65_plus_prop"] = np.round(
    p_static_agg["pop_65_plus"] / p_static_agg["pop_all_ages"], 3
)
  
p_static_agg["pop_white_prop"] = np.round(p_static_agg["white"] / p_static_agg["eth_all"], 3)
p_static_agg["pop_asian_prop"] = np.round(p_static_agg["asian"] / p_static_agg["eth_all"], 3)
p_static_agg["pop_mixed_prop"] = np.round(p_static_agg["mixed"] / p_static_agg["eth_all"], 3)
p_static_agg["pop_black_prop"] = np.round(p_static_agg["black"] / p_static_agg["eth_all"], 3)
p_static_agg["pop_other_prop"] = np.round(p_static_agg["other"] / p_static_agg["eth_all"], 3)
p_static_agg.drop( columns=["white", "asian", "mixed","black", "other"], inplace=True)

p_static_agg["noise"] = np.random.normal(0.0, 1.0, p_static_agg.shape[0])
p_static_agg_norm = p_static_agg.copy()

# %%
### This is where the time-varying feature aggregation happens:

p_timevarying = p_google_mobility.merge(p_rt[['date','la_name', 'Rt']], on=['date','la_name'])
p_timevarying.date = p_timevarying.date.astype(np.datetime64)

jan0101 = np.datetime64("2021-01-01")
# p_timevarying["yday"]=p_timevarying["date"].apply(lambda x:  (x - jan0101).days )
p_timevarying["ymonth"]=p_timevarying["date"].apply(lambda x: x.month )
p_timevarying["is_weekend"]=p_timevarying["date"].apply(lambda x:  int(x.weekday() >4) )


# %%
### This is where we merge all data together this IS a bit imperfect on purpose

p_all = p_timevarying.merge(p_static_agg, right_on='LAD21CD', left_on= 'lad19cd')
# p_all = p_all[p_all.LAD21CD.str.startswith('E') | p_all.LAD21CD.str.startswith('W')]

# %%
### This is where we select our explanatory variables  
feat_options = st.multiselect(
    "Which attributes groups should we use?" + 
    "(This model is not regularised or validated properly so it is not a good fit - used for illustration)",
    [
        "Age Proportions",
        "IMD",
        # "Space", # This is really strong in this small model// Needs proper regularisation
        "Population Sizes",
        "Ethnicity Proportions",  
    ],
)

option_age = "No"
option_imd = "No"
option_spa = "No"
option_pop = "No"
option_eth = "No" 

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

response_variable = ["Rt"]
mobility_features = ['retail_mob', 'transit_mob', 'workplace_mob', 'residential_mob']
temporal_features = ['is_weekend'  ]

cols_for_model =   mobility_features + temporal_features #+ ['noise']

if option_age == "Yes":
    cols_for_model = cols_for_model + [
        "pop_00_15_prop",
        "pop_16_29_prop",
        "pop_30_44_prop",
        "pop_45_64_prop",
        "pop_65_plus_prop",
    ]

if option_imd == "Yes":
    cols_for_model = cols_for_model + [
        "imd_full",
        "imd_health",
        "imd_employ",
        "imd_living",
    ]

if option_spa == "Yes":
    cols_for_model = cols_for_model + [
        "area",
        "lati",
        "long",
        "pop_density",
    ]

if option_pop == "Yes":
    cols_for_model = cols_for_model + [
        "pop_00_15",
        "pop_16_29",
        "pop_30_44",
        "pop_45_64",
        "pop_65_plus",
        "pop_all_ages",
    ]

if option_eth == "Yes":
    cols_for_model = cols_for_model + [
        "pop_white_prop",
        "pop_asian_prop",
        "pop_mixed_prop",
        "pop_black_prop",
        "pop_other_prop",
    ]


if (
    (option_spa == "Yes")
    | (option_imd == "Yes")
    | (option_age == "Yes")
    | (option_pop == "Yes") 
    | (option_eth == "Yes")
):
    pass # cols_for_model.remove("noise")

 
# %%    
### This is where we fit our regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

rf_reg = RandomForestRegressor(max_depth=6, random_state=0,n_estimators=20, n_jobs=-1)
full_df_work = p_all.dropna() # Obviously hammy!
X, y = full_df_work[cols_for_model], full_df_work[response_variable]

option_fit = st.selectbox("Ready to fit the model?", ["No", "Yes"]) 

if option_fit == "Yes":
    if (X.shape[1] == 5):
        st.write(f"Model fitted without any demographic information.")

    rf_reg.fit(X, y)  
    y_hat = rf_reg.predict(X)
    st.write(f"RMSE is: {mean_squared_error(y, y_hat) :.4f} and MAE {mean_absolute_error(y, y_hat) :.4f}. Are they any good? Who knows? They are an in-sample estimates so they are definitely optimistic.")
    st.write(f"The associated feature importance is:")
    st.write(pd.DataFrame( 
        zip(X.columns,rf_reg.feature_importances_), columns=['feat_name', 'feat_importance'] ).sort_values(
            "feat_importance",ascending=False))
    st.write("(In an ideal world we would show Partial Dependencies Plots here.)")

else: 
    y_hat = rng.normal( scale=10., loc=0., size=len(y))
# %%
full_df_work['hat_rt'] = y_hat
observed_n_expected_rt = full_df_work.groupby(['date', 'la_name'])[['Rt','hat_rt']].mean().reset_index()
observed_n_expected_rt.sort_values(['la_name','date'],inplace=True)

# %%
def make_scenario_df(_X:pd.DataFrame, retail_change:float=0.0, transit_change:float=0.0 , workplace_change:float=0.0 , residential_change:float=0.0 ):
    _X['retail_mob'] = _X['retail_mob'] + retail_change
    _X['transit_mob'] = _X['transit_mob'] + transit_change
    _X['workplace_mob'] = _X['workplace_mob'] + workplace_change
    _X['residential_mob'] = _X['residential_mob'] + residential_change
    return _X

user_retail_change = st.slider(
        "What is the change in retail mobility expected:", -30, 30, 0, 2
    )
user_transit_change = st.slider(
        "What is the change in transit mobility expected:", -30, 30, 0, 2
    )
user_workplace_change = st.slider(
        "What is the change in workplace mobility expected:", -30, 30, 0, 2
    )
user_residential_change = st.slider(
        "What is the change in residential mobility expected:", -30, 30, 0, 2
    )

lock_down_df = make_scenario_df(full_df_work.copy(), 
    retail_change = user_retail_change,
    transit_change = user_transit_change,  
    workplace_change = user_workplace_change, 
    residential_change=user_residential_change, 
)

if option_fit == "Yes":
    if( (user_retail_change==0) and (user_transit_change==0) and (user_workplace_change==0) and (user_residential_change==0)):
        st.write("No changes in Google mobility so no plot. :) ") 

    else:
        lock_down_df['y_hat_cf'] = rf_reg.predict(lock_down_df[X.columns])
        observed_n_counterfactual_rt = lock_down_df.groupby(['date', 'la_name'])[['Rt','y_hat_cf']].mean().reset_index()
        observed_n_counterfactual_rt.sort_values(['la_name','date'],inplace=True)

        C = observed_n_expected_rt.merge(observed_n_counterfactual_rt, on=['date', 'la_name'])
        C['the_effect'] = - C['hat_rt'] +  C['y_hat_cf']
        D = C[C.date >= "2021-11-01"][C.la_name.isin(['Bolton', 'Coventry', 'Derby','Rochford', 'Watford'])]

        _ymin = 0.975* np.min([D.hat_rt.min(),D.y_hat_cf.min()])
        _ymax = 1.025* np.max([D.hat_rt.max(),D.y_hat_cf.max()])

        mydates = [np.datetime64("2021-11-01"), np.datetime64("2021-11-08"), \
            np.datetime64("2021-11-15"), np.datetime64("2021-11-22"), np.datetime64("2021-11-29")]
        
        sns.set_style("darkgrid")
        fig, axs = plt.subplots(2, 2, figsize=(20, 13.33    ))  

        sfig1 = sns.lineplot(data=D, x='date', y='hat_rt', hue='la_name',ax=axs[0, 0])
        sfig1.set( xticks = mydates, ylim=(_ymin,_ymax))
        sfig2 = sns.lineplot(data=D, x='date', y='y_hat_cf', hue='la_name',ax=axs[0, 1])
        sfig2.set( xticks = mydates, ylim=(_ymin,_ymax))
        sfig3 = sns.lineplot(data=D, x='date', y='the_effect', hue='la_name',ax=axs[1, 0])
        sfig3.set( xticks = mydates)
        axs[0, 0].set_title('Expexted Rt for November 2021')
        axs[0, 1].set_title('Counterfactual Rt for November 2021')
        axs[1, 0].set_title('Expected differences for November 2021') 

        sns.kdeplot(data=D.groupby('la_name')['the_effect'].mean().reset_index(), 
                    x='the_effect', 
                    ax=axs[1, 1],)
        axs[1, 1].set_title('KDE plot of the average Expected difference of our policy on R(t) within LADs')
        axs[1, 1].set_ylabel('Density')
        axs[1, 1].set_xlabel('Average Expected difference within LAD')

        st.pyplot(fig)  
 

else:
    st.write("No model is yet fitted.")
# %%
