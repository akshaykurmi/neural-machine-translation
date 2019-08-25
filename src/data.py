import os
import io
import tarfile
from urllib.request import urlopen

import pandas as pd

from src import DATA_DIRECTORY


class EuroParlEnFr:
    URL = "http://statmt.org/europarl/v7/fr-en.tgz"
    FR = "europarl-v7.fr-en.fr"
    EN = "europarl-v7.fr-en.en"

    def load(self):
        with open(os.path.join(DATA_DIRECTORY, self.EN), "r") as f:
            en = f.readlines()
        with open(os.path.join(DATA_DIRECTORY, self.FR), "r") as f:
            fr = f.readlines()
        return pd.DataFrame(zip(en, fr), columns=["en", "fr"])

    def download(self):
        if all(os.path.exists(os.path.join(DATA_DIRECTORY, f)) for f in (self.FR, self.EN)):
            print("Data has already been downloaded.")
            return
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Downloading : {self.URL}")
        with urlopen(self.URL) as response:
            tf = tarfile.open(fileobj=io.BytesIO(response.read()))
        tf.extractall(path=DATA_DIRECTORY)
