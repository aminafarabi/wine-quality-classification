import pandas as pd
import numpy as np

def load_wine_data(red_path, white_path):
    red_wine = pd.read_csv(red_path, sep=';')
    white_wine = pd.read_csv(white_path, sep=';')

    red_wine["wine_type"] = 0      # red
    white_wine["wine_type"] = 1    # white

    df = pd.concat([red_wine, white_wine], axis=0)

    df["label"] = np.where(df["quality"] >= 6, 1, -1)

    X = df.drop(columns=["quality", "label"]).values
    y = df["label"].values

    return X, y
