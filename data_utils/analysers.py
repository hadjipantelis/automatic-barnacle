from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd

def get_knn_analysis_results(p_aggdata_normalised : pd.DataFrame, 
                             p_aggdata : pd.DataFrame, 
                             option_nn : int, 
                             option_pca : str, 
                             option_city : str,
                             cols_for_metric : list,
                             grouping_cols : list):
                             
    # Scale the input data to be N(0,1)
    cols_to_norm = p_aggdata_normalised.columns[2:]
    p_aggdata_normalised[cols_to_norm] = StandardScaler().fit_transform(
            p_aggdata_normalised[cols_to_norm]
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
    feat_matrix = p_aggdata_normalised[cols_for_metric]

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
    II = p_aggdata_normalised[option_city == p_aggdata_normalised[grouping_cols[1]]].index[0]

    # Add distances
    p_aggdata_normalised["distances"] = 0.0
    p_aggdata_normalised.loc[indices[II], "distances"] = _[II]

    map_data = p_aggdata.iloc[indices[II]][[grouping_cols[1], "lati", "long"]].reset_index(
        drop=True
    )
    map_data["geography_name"] = map_data[grouping_cols[1]]

    return map_data
