import pandas as pd
import numpy as np
import ads
from datetime import datetime

from drift_analysis import *

NAMESPACE = "frqap2zhtzbe"
BUCKET = "drift_input"
BUCKET_OUT = "drift_output"

REF_NAME = "reference.csv"
NEW_NAME = "current.csv"

print("Read dataset to compare and analyze...")

ref_url = f"oci://{BUCKET}@{NAMESPACE}/{REF_NAME}"
new_url = f"oci://{BUCKET}@{NAMESPACE}/{NEW_NAME}"

df_set1 = pd.read_csv(ref_url, storage_options={}, encoding="ISO-8859-1")
df_set2 = pd.read_csv(new_url, storage_options={}, encoding="ISO-8859-1")

# simulate a drift in set2
df_set2["MonthlyIncome"] = df_set2["MonthlyIncome"] + 2000
df_set2["Age"] = df_set2["Age"] + 5

# a single line of code
drifts = identify_data_drift(df_set1, df_set2, do_print=True, exc_list=["Attrition"])

print(drifts)

# save result as file
# if we want results as Pandas
result_df = pd.DataFrame(drifts)

now = datetime.now().strftime('%Y-%m-%d_%H_%M')

RESULT_FILE_NAME = f"drift_analysis_{now}.csv"

result_url = f"oci://{BUCKET_OUT}@{NAMESPACE}/{RESULT_FILE_NAME}"

print("Saving result file...")
result_df.to_csv(result_url, index=None)

print("Data Drift Analysis completed correctly!")


