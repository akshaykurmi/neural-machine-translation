import re
import os
import io
import json
import string
import shutil
import tarfile
import pickle as pkl
from urllib.request import urlopen

import requests
import pandas as pd
import tensorflow as tf
from google.cloud import storage


class EuroParlEnFr:
    URL = "http://statmt.org/europarl/v7/fr-en.tgz"
    FR = "europarl-v7.fr-en.fr"
    EN = "europarl-v7.fr-en.en"

    def __init__(self, data_directory):
        self.data_directory = data_directory

    def load(self):
        self._download()
        with open(os.path.join(self.data_directory, self.EN), "r") as f:
            en = f.readlines()
        with open(os.path.join(self.data_directory, self.FR), "r") as f:
            fr = f.readlines()
        return pd.DataFrame(zip(en, fr), columns=["en", "fr"])

    def _download(self):
        if all(os.path.exists(os.path.join(self.data_directory, f)) for f in (self.FR, self.EN)):
            print("Data has already been downloaded.")
            return
        os.makedirs(self.data_directory, exist_ok=True)
        print(f"Downloading : {self.URL}")
        with urlopen(self.URL) as response:
            f = tarfile.open(fileobj=io.BytesIO(response.read()))
        f.extractall(path=self.data_directory)


class ManyThingsEnFr:
    URL = "http://www.manythings.org/anki/fra-eng.zip"
    FILE = "manythingsenfr.tsv"

    def __init__(self, data_directory):
        self.data_directory = data_directory

    def load(self):
        if not os.path.exists(os.path.join(self.data_directory, self.FILE)):
            raise FileNotFoundError(f"{self.FILE} unavailable. Download file from {self.URL}")
        return pd.read_csv(os.path.join(self.data_directory, self.FILE), sep="\t", names=["en", "fr"])


def preprocess_sentence(s):
    s = s.lower()
    s = re.sub(f'([{string.punctuation}])', r' \1 ', s)
    s = re.sub(f'\s+', r' ', s)
    return "<sos> " + s.strip() + " <eos>"


def create_tokenizer_and_preprocessed_files(dataset, data_directory, prefix, overwrite=False):
    def create_tokenizer(sentences):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<unk>")
        tokenizer.fit_on_texts(sentences)
        return tokenizer

    if not overwrite and all([os.path.exists(os.path.join(data_directory, f"{prefix}.{f}")) for f in
                              ["preprocessed.csv", "en_tokenizer.pkl", "fr_tokenizer.pkl"]]):
        return

    df = dataset.load()
    df["en"] = df["en"].apply(preprocess_sentence)
    df["fr"] = df["fr"].apply(preprocess_sentence)
    df.to_csv(os.path.join(data_directory, f"{prefix}.preprocessed.csv"), sep="|", index=False)
    df["en"] = df["en"].apply(lambda s: s.split())
    df["fr"] = df["fr"].apply(lambda s: s.split())
    with open(os.path.join(data_directory, f"{prefix}.en_tokenizer.pkl"), "wb") as en_t:
        pkl.dump(create_tokenizer(df["en"]), en_t)
    with open(os.path.join(data_directory, f"{prefix}.fr_tokenizer.pkl"), "wb") as fr_t:
        pkl.dump(create_tokenizer(df["fr"]), fr_t)


def load_tokenizers(data_directory, prefix):
    with open(os.path.join(data_directory, f"{prefix}.en_tokenizer.pkl"), "rb") as en_t, open(
            os.path.join(data_directory, f"{prefix}.fr_tokenizer.pkl"), "rb") as fr_t:
        return pkl.load(en_t), pkl.load(fr_t)


def create_tf_records(data_directory, prefix, overwrite=False):
    def make_sequence_example(en, fr):
        ex = tf.train.SequenceExample()
        en_feature = ex.feature_lists.feature_list["en"]
        fr_feature = ex.feature_lists.feature_list["fr"]
        for token in en:
            en_feature.feature.add().int64_list.value.append(token)
        for token in fr:
            fr_feature.feature.add().int64_list.value.append(token)
        return ex

    if not overwrite and all([os.path.exists(os.path.join(data_directory, f"{prefix}.{f}"))
                              for f in ["tfrecord", "metadata.json"]]):
        return

    en_tokenizer, fr_tokenizer = load_tokenizers(data_directory, prefix)
    num_examples = 0
    with tf.io.TFRecordWriter(os.path.join(data_directory, f"{prefix}.tfrecord")) as writer:
        for i, chunk in enumerate(
                pd.read_csv(os.path.join(data_directory, f"{prefix}.preprocessed.csv"), sep="|", chunksize=100000)):
            num_examples += chunk.shape[0]
            print(f"Processing chunk {i}, {num_examples}")
            chunk["en"] = chunk["en"].apply(lambda s: s.split())
            chunk["fr"] = chunk["fr"].apply(lambda s: s.split())
            chunk["en"] = en_tokenizer.texts_to_sequences(chunk["en"])
            chunk["fr"] = en_tokenizer.texts_to_sequences(chunk["fr"])
            for _, row in chunk.iterrows():
                writer.write(make_sequence_example(row['en'], row['fr']).SerializeToString())
    with open(os.path.join(data_directory, f"{prefix}.metadata.json"), "w") as f:
        json.dump({"num_examples": num_examples}, f)


def load_datasets(batch_size, data_directory, prefix):
    def parse_example_proto(ex):
        sequence_features = {
            "en": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
            "fr": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        _, sequence = tf.io.parse_single_sequence_example(ex, sequence_features=sequence_features)
        return sequence["en"], sequence["fr"]

    with open(os.path.join(data_directory, f"{prefix}.metadata.json"), "r") as f:
        num_examples = json.load(f)["num_examples"]
    dataset = tf.data.TFRecordDataset(filenames=[os.path.join(data_directory, f"{prefix}.tfrecord")])
    dataset = dataset.map(parse_example_proto)
    dataset = dataset.shuffle(buffer_size=num_examples)
    split = (num_examples * 95) // 100
    dataset_train = dataset.take(split).padded_batch(batch_size, padded_shapes=([None], [None]), drop_remainder=False)
    dataset_val = dataset.skip(split).padded_batch(batch_size, padded_shapes=([None], [None]), drop_remainder=False)
    return dataset_train, dataset_val


def zip_directory_and_upload_to_gcs(directory, bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete() if blob.exists() else None
    name, ext = os.path.splitext(blob_name)
    shutil.make_archive(name, ext[1:], directory)
    url = blob.create_resumable_upload_session()
    print(f"Uploading {blob_name} to {bucket_name} ...")
    with open(blob_name, "rb") as f:
        requests.put(url, data=f, headers={"Content-type": 'application/octet-stream'})
    os.remove(blob_name)


def download_from_gcs_and_unzip_directory(directory, bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        print(f"Blob {blob_name} not found in {bucket_name}.")
        return
    print(f"Downloading {blob_name} from {bucket_name} ...")
    blob.download_to_filename(blob_name)
    shutil.unpack_archive(blob_name, directory)
    os.remove(blob_name)
