# %%
from sklearn.linear_model import ElasticNetCV
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from pandas.core.common import flatten

st.title('How is my LTLA doing since February in terms of cases per 100K?')
st.write("A synthetic controls inspired view.")

# %%
# Read LAD boundaries (to have Lat-Long)
# Data from:
# https://prod-hub-indexer.s3.amazonaws.com/files/1d78d47c87df4212b79fe2323aae8e08/0/full/27700/1d78d47c87df4212b79fe2323aae8e08_0_full_27700.csv
p_lads_location = pd.read_csv(
    "data/Local_Authority_Districts_(December_2019)_Boundaries_UK_BFC.csv")
p_lads_location.drop(["st_lengthshape", "bng_n", "bng_e",
                      "lad19nmw", "objectid"], axis=1, inplace=True)

# %%
# Data from:
# https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland/mid2019april2020localauthoritydistrictcodes/ukmidyearestimates20192020ladcodes.xls
p_lads_popul = pd.read_excel(
    "data/ukmidyearestimates20192020ladcodes-1.xls",
    sheet_name="MYE2 - Persons",
    skiprows=3)
p_lads_popul.columns = p_lads_popul.iloc[0, :]
p_lads_popul = p_lads_popul.iloc[1:, :]
p_lads_popul["pop_00_15"] = p_lads_popul[np.arange(0, 16)].sum(axis=1)
p_lads_popul["pop_16_75"] = p_lads_popul[np.arange(16, 76)].sum(axis=1)
p_lads_popul["pop_76_90"] = p_lads_popul[[
    "90+"] + list(np.arange(76, 90))].sum(axis=1)
p_lads_popul["pop_total"] = p_lads_popul["pop_00_15"] + \
    p_lads_popul["pop_16_75"] + p_lads_popul["pop_76_90"]

p_lads_popul["pop_00_15_prop"] = p_lads_popul["pop_00_15"] / \
    p_lads_popul["pop_total"]
p_lads_popul["pop_16_75_prop"] = p_lads_popul["pop_16_75"] / \
    p_lads_popul["pop_total"]
p_lads_popul["pop_76_90_prop"] = p_lads_popul["pop_76_90"] / \
    p_lads_popul["pop_total"]

p_lads_popul["age_mean"] = np.dot(p_lads_popul[list(
    np.arange(0, 90))], np.arange(1, 91)) / p_lads_popul["pop_total"]

# %%
# Read COVID-19 cases per LAD
# https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv
# (March 6, 2021)
p_lads_cases = pd.read_csv(
    "https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv")

p_lads_cases = p_lads_cases[p_lads_cases['Area type'] == 'ltla']
p_lads_cases = p_lads_cases.sort_values(['Area code', 'Specimen date', ])
p_lads_cases = p_lads_cases.merge(
    p_lads_popul[['pop_total', 'Code']], right_on="Code", left_on="Area code")

p_lads_cases['roll_daily_confirmed'] = p_lads_cases.groupby('Area code').rolling(
    7)['Daily lab-confirmed cases'].mean().reset_index(drop=True)

p_lads_cases['roll_cases_100k'] = p_lads_cases['roll_daily_confirmed'] / \
    p_lads_cases['pop_total'] * (100_000)

p_lads_cases["Specimen date"] = p_lads_cases["Specimen date"].astype(
    np.datetime64)
p_lads_cases = p_lads_cases[p_lads_cases["Specimen date"]
                            >= "2020-09-01"].copy()

# %%
# Combine all the statistic predictors together (just Lat-Long &
# Population here)
p_lads_static = p_lads_location.merge(p_lads_popul[[
    "Code", "Name", "age_mean",
    "pop_total", "pop_00_15", "pop_16_75", "pop_76_90",
    "pop_00_15_prop", "pop_16_75_prop", "pop_76_90_prop"]], right_on="Code", left_on="lad19cd")
p_lads_static = p_lads_static[p_lads_static.lad19cd.str.startswith('E')].copy()

code_to_name_dict = dict(zip(p_lads_static.lad19cd, p_lads_static.lad19nm))
name_to_code_dict = dict(zip(p_lads_static.lad19nm, p_lads_static.lad19cd))
# Drop Isle of Scilly and City of London
p_lads_static.drop(p_lads_static[p_lads_static.lad19cd.isin(
    ['E06000053', 'E09000001'])].index, inplace=True)
p_lads_static.reset_index(inplace=True)
del p_lads_location, p_lads_popul

# %%
# Make response variable covariates
p_lads_cases["ymonth"] = p_lads_cases["Specimen date"].apply(lambda x: x.month)
p_lads_cases_agg = p_lads_cases.groupby(["ymonth", "Area code"])["roll_cases_100k"].apply(
    lambda x: list(flatten([np.mean(x), np.quantile(x, [0.25, 0.5, 0.75])]))).reset_index()

p_lads_cases_agg = pd.concat([pd.DataFrame(p_lads_cases_agg["roll_cases_100k"].values.tolist(
)), p_lads_cases_agg[["ymonth", "Area code"]]], axis=1)

p_lads_cases_agg.rename({
    0: "roll_cases_100k_mean",
    1: "roll_cases_100k_q25",
    2: "roll_cases_100k_q50",
    3: "roll_cases_100k_q75"
}, inplace=True, axis=1)

p_lads_cases_pivot = pd.pivot_table(p_lads_cases_agg[(p_lads_cases_agg.ymonth > 4) |
                                                     (p_lads_cases_agg.ymonth == 1)],
                                    values=[
    'roll_cases_100k_q25',
    'roll_cases_100k_q50',
    'roll_cases_100k_q75',
    # 'roll_cases_100k_mean'
],
    index=['Area code', ],
    columns=['ymonth', ]).reset_index().sort_values(['Area code'])


# %%
option_city = st.selectbox(
    'What is your LTLA (Lower Tier Local Authority):',
    np.sort(
        p_lads_static.lad19nm))

# %%
# Make query LAD
# query_lad_code="E06000021" # Stoke
# query_lad_code="E07000108" # Dover
# query_lad_code = "E07000103" # Watford
query_lad_code = name_to_code_dict[option_city]


# Remove the 10 closest geographical LADs (and the query LAD itself) from
# the donor pool

option_nn = st.selectbox(
    'How many of the nearest LADs to remove:',
    np.arange(
        5,
        56,
        5))
st.write(
    f"We will exclude {option_nn} of {option_city}'s nearest neighbors from the synthetic control donor pool. " +
    f"Usually the larger the exclusion list, the more robust the results but also less likely to fit well.")

k = option_nn + 1

nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',)
nbrs.fit(p_lads_static[["lat", "long"]])
_, indices = nbrs.kneighbors(p_lads_static[["lat", "long"]])

II = p_lads_static[query_lad_code == p_lads_static.lad19cd].index[0]
map_data = p_lads_static.iloc[indices[II]][[
    'lad19cd', "lat", "long"]].reset_index(drop=True)
del II, nbrs, k

y = p_lads_cases_pivot[(p_lads_cases_pivot["Area code"] == query_lad_code)] 
X = p_lads_cases_pivot[[ladcd not in list(
    map_data.lad19cd) for ladcd in p_lads_cases_pivot["Area code"]]]
y_train = y.drop(["Area code"], axis=1).copy().transpose()
X_train = X.drop(["Area code"], axis=1).copy().transpose()

st.write(
    f"Methodology-wise:  " +
    f"\n 1. We have {X_train.shape[1]} LADs in our donor pool.  (We have {p_lads_static.shape[0]-1} LADs availabe in total minus the exclusion list.)  " +
    f"\n 2. We use the per month average rate, 25th, 50th (median) and 75th quantile of the cases per 100K as the features we care for.  " +
    f"\n 3. We find the linear combination W of them that minimises the discremancy X_j - X_0*W. " +
    f"\n - X_j is the k-length vector containing values of the pre-intervention features of the query unit j." +
    f"\n - X_0 is the k-by-{X_train.shape[1]} matrix containing values of the pre-intervention features of the donor pool units." +
    f"\n 4. We minimize this discremancy by using an Elastic Net solver. We set all the coefficients to be non-negative and we have no intercept.  "
    f"\n 5. We create the synthetic control by multiplying the observed trajectories from the donor list with the coeffecients found above by Elastic Net.")

# %%
# Train learner
learner = ElasticNetCV(
    l1_ratio=[
        0.01,
        0.05,
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
        0.95,
        0.99,
        1.0],
    cv=10,
    verbose=False,
    fit_intercept=False,
    positive=True,
    random_state=3,
    selection='random',
    n_alphas=101)
learner.fit(X_train, y_train)

# %%
# Make synthetic control
p_lads_cases_mult = pd.pivot_table(p_lads_cases[['Area code', 'Specimen date', 'roll_cases_100k']],
                                   values=['roll_cases_100k'],
                                   index=['Area code', ],
                                   columns=['Specimen date', ]).reset_index()
# Keep only LADs used
p_lads_cases_mult = p_lads_cases_mult[[ladcd not in list(
    map_data.lad19cd) for ladcd in p_lads_cases_mult["Area code"]]].sort_values(['Area code'])

p_lads_cases_mult.columns = ['_'.join(list(map(str, col))).strip(
    '_') for col in p_lads_cases_mult.columns.values]
X_est = p_lads_cases_mult.drop(['Area code'], axis=1)
synthetic_lad_trajectory = np.dot(learner.coef_, X_est)


# %%
# Somw quick plots
end_date = p_lads_cases["Specimen date"].max()
start_date = p_lads_cases["Specimen date"].min()
plt.rcParams["figure.figsize"] = (15, 7.5)
fig, ax = plt.subplots()

np.random.seed(4)
max_y = 0.0
for lad in np.random.choice(p_lads_cases_mult['Area code'], 12):
    p_df = p_lads_cases_mult[p_lads_cases_mult['Area code'] == lad]
    p_df.drop(['Area code'], axis=1, inplace=True)
    p_df_2 = p_df.transpose().reset_index()
    p_df_2['time'] = pd.date_range(start_date, end_date)
    p_df_2.columns = ["index", "roll_cases_100k", "time"]
    sns.lineplot(data=p_df_2, label=code_to_name_dict[lad], ax=ax,
                 x="time", y='roll_cases_100k', alpha=0.5)
    if max_y < np.max(p_df_2.roll_cases_100k):
        max_y = 1.025 * np.max(p_df_2.roll_cases_100k)

ax.plot(pd.date_range(start_date, end_date),
        p_lads_cases[p_lads_cases['Area code'] ==
                     query_lad_code].roll_cases_100k, c='blue',
        label='Observed ' + code_to_name_dict[query_lad_code])
ax.plot(pd.date_range(start_date, end_date), synthetic_lad_trajectory,
        c='black', label='Synthetic ' + code_to_name_dict[query_lad_code])
ax.set_xlabel('Time', fontweight='bold')
ax.set_ylabel('Cases per 100k (7-day rolling mean)', fontweight='bold')

for dd in pd.date_range(start_date, end_date, freq="MS"):
    ax.plot([dd, dd], [0, max_y], c='grey', alpha=0.33, linestyle='dashed')

for yy in np.arange(0, max_y, 25):
    ax.plot([start_date, end_date], [yy, yy],
            c='grey', alpha=0.33, linestyle='dashed')

ax.plot([pd.to_datetime("2021-02-01")] * 2, [0, max_y],
        c='red', alpha=0.66, linestyle='dashed')

ax.legend()
st.pyplot(fig)

st.write(
    f"We show observed and synthetic {option_city} as well as dozen random LAD trajectories. " +
    f"If our observed LAD trajectory (solid blue line) from February onwards is consistently below the synthetic trajecory " +
    f"(solid black line) then it seems {option_city} is doing better than expected, if it is consistently above the synthetic " +
    f"trajecory then {option_city} does worse than expected.")


# %%
# Make the synthetic control donor pool used
Z = X.iloc[(learner.coef_ > 0), :][["Area code"]]
Z['betas'] = learner.coef_[learner.coef_ > 0]
Z['Area name'] = Z["Area code"].apply(lambda x: code_to_name_dict[x])
Z['Normalised weight'] = Z['betas'] / Z['betas'].sum()

# %%
st.write("These are are the LADs used:")


# %%
st.write(Z[["Area name", "Normalised weight"]].reset_index(
    drop=True).sort_values(["Normalised weight"], ascending=False))
# %%
