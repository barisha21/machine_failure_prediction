# =====================================================
# IMPORT LIBRARY
# =====================================================

import pandas as pd


# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset(path):

    df = pd.read_csv(path)

    print("Dataset Loaded Successfully!")

    print("Rows :", df.shape[0])
    print("Columns :", df.shape[1])

    return df