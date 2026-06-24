import os
import tarfile
import pandas as pd
from urllib.request import urlretrieve


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
HOUSING_PATH = os.path.join("data", "housing")

def fetch_housing_data(
  housing_url = HOUSING_URL,
  housing_path = HOUSING_PATH      
):
    os.makedirs(housing_path, exist_ok=True)

    tgz_path = os.path.join(housing_path, "housing.tgz")

    print("Downloading dataset.....")
    urlretrieve(housing_url, tgz_path)

    print("Extracting dataset.....")
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

    print("Done!")

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)