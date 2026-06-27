import pandas as pd


def clean_data(df):

    # Remove duplicate records
    df = df.drop_duplicates()

    # Drop unnecessary columns
    df.drop(
        [
            "UDI",
            "Product ID",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF"
        ],
        axis=1,
        inplace=True
    )

    return df