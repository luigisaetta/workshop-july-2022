import pandas as pd
import numpy as np
import os
import tempfile

# to save/load from Model Catalog
from ads.catalog.model import ModelCatalog
from ads.common.model_metadata import UseCaseType, MetadataCustomCategory
from ads.model.generic_model import GenericModel

from scipy.stats import ks_2samp, chisquare, chi2_contingency, gaussian_kde
from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
import seaborn as sns

#
# if we want eclude some columns (ex: target, we can use exc_list)
#
def identify_categorical(df, min_distinct=10, exc_list=[]):
    # if you pass the TARGET it will be excluded from the features list
    cat_columns = []
    
    # remove exclusione list
    col_to_analyze = list(set(df.columns) - set(exc_list))
    
    for col in col_to_analyze:
        # identifichiamo come categoriche tutte le colonne che soddisfano questa condizione !!!
        # la soglia la possiamo cambiare (parm. min_distinct)
        if df[col].dtypes == 'object' or df[col].nunique() < min_distinct:
            cat_columns.append(col)
            
    return cat_columns

# (Wikipedia) Pearson's chi-squared test is used to determine whether there is a statistically significant difference 
# between the expected frequencies and the observed frequencies in one or more categories of a contingency table.

# contingency table contains the occurrencies for each value in reference and current set
# contiene il conteggio delle occorrenze del primo dataset e nel secondo
def compute_contingency_table(ref_col, newset_col):
    # we expect here Pandas df cols (ex: df_ref[col])
    # see https://github.com/Azure/data-model-drift/blob/main/tabular-data/utils.py
    
    index = list(set(ref_col.unique()) | set(newset_col.unique()))

    # this is the best way to handle value missing in one of the two
    value_counts_df = pd.DataFrame(ref_col.value_counts(), index=index)
    value_counts_df.columns = ['reference']
    value_counts_df['new'] = newset_col.value_counts()
    # aggiunge 0 per valori mancanti in uno dei dataset
    value_counts_df.fillna(0, inplace=True)

    result = np.array([[value_counts_df['reference'].values], [value_counts_df['new'].values]])
    
    return result, index

# we're using scipy functions (see import)

# min. number of rows to be considered numerical (not categorical)

# we can specify a list of cols to exclude from drift analysis
def identify_data_drift(df_ref, df_new, p_thr=0.01, do_print=True, exc_list=[]):
    all_cols = df_ref.columns
    
    # p_thr is the threshold used for Null Hyp. tests
    
    # check that the two dataframe have the same columns
    
    if list(df_ref.columns) != list(df_new.columns):
        print("Error: The two DataFrame must have the same columns.")
        print("Closing.")
        
        return None
    
    # identify categorical and numerical columns
    cat_cols = identify_categorical(df_ref, exc_list=exc_list)
    # all the rest excluding target
    num_cols = list(set(all_cols) - set(cat_cols) - set(exc_list))
    
    if do_print:
        print()
        print("*** Report on evidences of Data Drift identified ***")
        print()
    
    # compute only once the describe, to get stats
    # this way is faster!
    set1_describe = df_ref.describe().T
    set2_describe = df_new.describe().T
    
    # enforce exclusion list
    col_to_analyze = sorted(list(set(df_ref.columns) - set(exc_list)))
    
    list_drifts = []
    for col in col_to_analyze:
        # per prendere solo le numeriche
        if col in num_cols:
            # analyze numerical columns using Kolmogoros Smirnov test
            stats, p_value = ks_2samp(df_ref[col].values, df_new[col].values)
            
            type = "continuous"
            
            # save also the stats for the column (see df.describe().T)
            # I have reduced the n. of digits to 2
            # we don't take the count
            stats1 = str(list(np.round(set1_describe.loc[col, :].values[1:], 2))) 
            stats2 = str(list(np.round(set2_describe.loc[col, :].values[1:], 2)))
            stats = stats1 + "," + stats2
            
            # compute the wasserstein distance
            was_distance = wasserstein_distance(df_ref[col].values, df_new[col].values)
            # normalize with mean
            was_distance = round(was_distance/(set1_describe.loc[col, 'mean']), 3)
            
            # compute mean difference/mean1
            mean1 = set1_describe.loc[col, 'mean']
            mean2 = set2_describe.loc[col, 'mean']
            delta_mean_norm = round(abs(mean1 - mean2)/mean1, 3)
            
        # solo le categoriche
        if col in cat_cols:
            # compute table with occurrencies
            c_table, index = compute_contingency_table(df_ref[col], df_new[col])

            stats, p_value, dof, _ = chi2_contingency(c_table)
            
            type = "categorical"
            stats = str(c_table)
            
        # p_value can be interpreted as the probability that the two dataset come from the same distribution
        # if it is too low.. they DON'T
        if p_value < p_thr:
            # detected drift
            p_val_rounded = round(p_value, 5)
            
            if do_print:
                print("Identified drift in column:", col)
                print(f"p_value: {p_val_rounded}")
                print()
            
            drift_info = {"Column": col, 
                          "Type" : type, 
                          "p_value": p_val_rounded,
                          "threshold" : p_thr,
                          "stats" : stats
                         }
            if type == "continuous":
                drift_info["was_distance_norm"] = was_distance
                drift_info["delta_mean_norm"] = delta_mean_norm
                
            list_drifts.append(drift_info)
    
    if (len(list_drifts) == 0):
        if do_print:
            print("No evidence found.")
            print()
        
    return list_drifts

#
# functions for plotting
#

# make a plot to compare the two distribution
# only for numerical
def plot_comparison_numeric(dfset1, dfset2, col):
    df1_dict= {col: list(dfset1[col].values)}
    df1 = pd.DataFrame(df1_dict)

    df2_dict= {col: list(dfset2[col].values)}
    df2 = pd.DataFrame(df2_dict)

    df12 = pd.concat([df1.assign(dataset='set1'), df2.assign(dataset='set2')], ignore_index=True)
    
    plt.figure(figsize=(9,6))
    plt.title(f"Distribution comparison for {col}")
    sns.histplot(data=df12, x=col, hue="dataset", kde=True)
    plt.grid(True)
    plt.show();
    
def plot_comparison_categorical2(dfset1, dfset2, col):
    # compute the contingency table
    c_table, index = compute_contingency_table(dfset1[col], dfset2[col])
    
    # occ set1
    occ1 = c_table[0, 0, :]
    # set2
    occ2 = c_table[1, 0, :]
    
    ROWS = 1
    COLS = 2
    
    plt.figure(figsize=(12, 6))
    for i, occ in enumerate([occ1, occ2]):
        plt.subplot(ROWS, COLS, i+1)
        plt.title(f"Distribution for set{i+1}")
        sns.barplot(x=index, y=occ, color="r")
        plt.grid(True)
    plt.show()
    
def plot_comparison_categorical1(dfset1, dfset2, col):
    # compute the contingency table
    c_table, index = compute_contingency_table(dfset1[col], dfset2[col])
    
    # occ set1
    occ1 = c_table[0, 0, :]
    # set2
    occ2 = c_table[1, 0, :]
    
    # create a dataframe with two columns
    dict_df = {"set1" : list(occ1), "set2" : list(occ2)}
    
    # add index
    df_data = pd.DataFrame(dict_df, index=index)
    
    ax = df_data.plot.bar(rot=30, 
                          # color=['green', 'blue'], 
                          figsize=(9, 6),
                         title=f"Distribution comparison for {col}")
    
    plt.grid(True)
    plt.show()
    
#
# get the reference dataset URL from Model Catalog
#
def get_reference_dataset_url(model_ocid):
    KEY_REF_DS = "reference dataset"
    
    # load model from Model Catalog
    # for reading custom metadata I can use GenericModel

    # if you want to use the model, better to use a specific class (ex: SklearnModel) to avoid two access
    gen_model = GenericModel.from_model_catalog(model_id=model_ocid,
                                                # only for temporary use
                                                model_file_name="gen_model.pkl",
                                                artifact_dir=tempfile.mkdtemp())
    
    # take the custom metadata as Pandas df
    meta_df = gen_model.metadata_custom.to_dataframe()
    
    # get only one row
    condition = (meta_df['Key'] == KEY_REF_DS)
    ref_ds_url_arr = meta_df.loc[condition]['Value']
    
    # it is a np array... take the only row
    ref_url = None
    # checck that it has found something
    if (ref_ds_url_arr is not None) and (ref_ds_url_arr.shape[0] > 0):
        ref_url = ref_ds_url_arr.values[0]
        
    return ref_url