import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

# Data Manipulation, Preprocessing and Cleaning
import json, yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
)

# Sentence pre-processing
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download the NLTK tokenizer models (if not already downloaded)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

# Linear Algebra
import numpy as np

# Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Data Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import transformers
from transformers import pipeline

# ML evaluation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Saving the Model
import joblib

# Downloading
import requests

# For Best Coding Practices
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Dict, List, Any, Union
from dataclasses import dataclass
from box.exceptions import BoxValueError


### Helper Functions
# Stemming
def stem_sentence(sentence):
    stemmer = PorterStemmer()
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(stemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


## Lemmatization
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = " ".join(
        [
            lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in nltk.word_tokenize(sentence)
        ]
    )
    return lemmatized_output
