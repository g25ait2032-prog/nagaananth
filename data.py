"""
data.py — 
Data loading, sampling, train/test split, and tokenizer encoding
MLOps Assignment 2 | IIT Jodhpur
"""

import gzip
import json
import pickle
import random
import requests

from transformers import DistilBertTokenizerFast

from utils import build_label_maps, MyDataset


# ── Data source URLs ─────────────────────────────────────────────────────────
GENRE_URL_DICT = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


def load_reviews(url: str, head: int = 10000, sample_size: int = 2000) -> list[str]:
    """Stream reviews from a gzipped JSON-lines URL and return a random sample."""
    reviews = []
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(response.raw, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if head is not None and i >= head:
                break
            d = json.loads(line)
            reviews.append(d["review_text"])
    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres(
    genre_url_dict: dict = GENRE_URL_DICT,
    pickle_path: str = "genre_reviews_dict.pickle",
    head: int = 10000,
    sample_size: int = 2000,
) -> dict:
    """
    Download (or load from cache) reviews for every genre.
    Saves/loads a pickle file to avoid re-downloading on reruns.
    """
    try:
        print(f"Loading cached data from {pickle_path} ...")
        return pickle.load(open(pickle_path, "rb"))
    except FileNotFoundError:
        pass

    genre_reviews_dict = {}
    for genre, url in genre_url_dict.items():
        print(f"Downloading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url, head=head, sample_size=sample_size)

    pickle.dump(genre_reviews_dict, open(pickle_path, "wb"))
    print(f"Saved to {pickle_path}")
    return genre_reviews_dict


def train_test_split(
    genre_reviews_dict: dict,
    reviews_per_genre: int = 1000,
    train_frac: float = 0.8,
) -> tuple[list, list, list, list]:
    """
    Split genre reviews into train/test lists of texts and string labels.

    Args:
        genre_reviews_dict: {genre: [review_text, ...]}
        reviews_per_genre:  how many reviews to use per genre
        train_frac:         fraction used for training

    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        split = int(len(sampled) * train_frac)
        for text in sampled[:split]:
            train_texts.append(text)
            train_labels.append(genre)
        for text in sampled[split:]:
            test_texts.append(text)
            test_labels.append(genre)

    print(
        f"Train size: {len(train_texts)}  |  Test size: {len(test_texts)}  |  "
        f"Genres: {len(genre_reviews_dict)}"
    )
    return train_texts, train_labels, test_texts, test_labels


def encode_datasets(
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    tokenizer: DistilBertTokenizerFast,
    max_length: int = 512,
) -> tuple[MyDataset, MyDataset]:
    """
    Tokenize texts and encode string labels to integers.
    Returns PyTorch Dataset objects ready for the Trainer.
    """
    label2id, _ = build_label_maps(train_labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=max_length)

    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc  = [label2id[y] for y in test_labels]

    return (
        MyDataset(train_encodings, train_labels_enc),
        MyDataset(test_encodings,  test_labels_enc),
    )


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from transformers import DistilBertTokenizerFast

    MODEL_NAME = "distilbert-base-cased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    genre_reviews = load_all_genres()
    tr_texts, tr_labels, te_texts, te_labels = train_test_split(genre_reviews)
    train_ds, test_ds = encode_datasets(tr_texts, tr_labels, te_texts, te_labels, tokenizer)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test  dataset size: {len(test_ds)}")
    print("data.py ran successfully.")
