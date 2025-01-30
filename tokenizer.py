import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import string
import re

df = pd.read_csv('dataset/english_to_german.csv')

df.columns = df.columns.str.strip()

df['source'] = df['German']
df['target'] = df['English'].apply(lambda x: '[start] ' + x + ' [end]')

df = df.drop(['English', 'German'], axis=1)

print(df.sample(8))

# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# split the data into train, validation, and test sets
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.2)
test_size = int(len(df) * 0.1)

train_df = df[:train_size]
val_df = df[train_size:train_size+val_size]
test_df = df[train_size+val_size:]

# Standardizing, tokenizing and indexing the data
max_tokens = 25000
sequence_length = 30

strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
 
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

# tokenize the data using our custom standardization function
source_vectorization = keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length + 1, # add +1 token to our target sentences since they'll be shifted right by 1 during training
    standardize=custom_standardization,
)

# index all tokens in the source and target sentences
train_source_texts = train_df['source'].values
train_target_texts = train_df['target'].values
source_vectorization.adapt(train_source_texts)
target_vectorization.adapt(train_target_texts)

source_vectorization_model = keras.Sequential([source_vectorization])
target_vectorization_model = keras.Sequential([target_vectorization])

source_vectorization_model.save("tokens_de.h5")
target_vectorization_model.save("tokens_en.h5")

