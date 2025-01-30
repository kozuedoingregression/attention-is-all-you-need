import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, Dropout


import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt

df = pd.read_csv('dataset/german_to_english.csv')
print(df.head(8))

german_sentences = df["German"].tolist()
english_sentences = df['English'].tolist()

print("German sentences \n",german_sentences)
print("English sentences \n",english_sentences)

tokenizer_de = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    german_sentences, target_vocab_size = 2**15
)

tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    english_sentences, target_vocab_size = 2**15
)

tokenizer_de.save_to_file("tokenizer_de")
tokenizer_en.save_to_file("tokenizer_en")

# added a special token for start and end of the sentences
START_TOKEN = [tokenizer_en.vocab_size]
END_TOKEN = [tokenizer_en.vocab_size + 1]

def encode(de_sentence, en_sentence):
    de_encoded = START_TOKEN + tokenizer_de.encode(de_sentence) + END_TOKEN
    en_encoded = START_TOKEN + tokenizer_en.encode(en_sentence) + END_TOKEN
    return de_encoded, en_encoded

encoded_data = []

for de,en in zip(german_sentences, english_sentences):
    de_encoded, en_encoded = encode(de,en)
    encoded_data.append((de_encoded, en_encoded))

# split into seperate lists
german_encoded, english_encoded = zip(*encoded_data)

#english_encoded

max_len_de = max(len(seq) for seq in german_encoded)
max_len_en = max(len(seq) for seq in english_encoded)

# Pad sequences
german_padded = tf.keras.preprocessing.sequence.pad_sequences(
    german_encoded, maxlen=max_len_de, padding='post', truncating='post'
)
english_padded = tf.keras.preprocessing.sequence.pad_sequences(
    english_encoded, maxlen=max_len_en, padding='post', truncating='post'
)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((
    {'encoder_inputs': german_padded, 'decoder_inputs': english_padded[:, :-1]},  # Shift right
    {'outputs': english_padded[:, 1:]}  # Shift left (target)
))

# Batch and prefetch
BUFFER_SIZE = 20000
BATCH_SIZE = 64
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)  # Ensure d_model is float32

    def get_angles(self, position, i):
        # Cast all terms to float32 explicitly
        i = tf.cast(i, tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / self.d_model)
        return tf.cast(position, tf.float32) * angle_rates

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]

        # Generate angles with float32 dtypes
        angles = self.get_angles(
            tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        )

        # Concatenate sine and cosine parts
        sine_part = tf.sin(angles[:, 0::2])
        cosine_part = tf.cos(angles[:, 1::2])
        angles = tf.concat([sine_part, cosine_part], axis=-1)

        # Add to embeddings (broadcasted across batch dimension)
        return inputs + angles[tf.newaxis, ...]

    def compute_output_spec(self, inputs):
        # Explicitly define output shape/dtype for Keras
        return tf.TensorSpec(inputs.shape, inputs.dtype)

# Test positional encoding
pe = PositionalEncoding(512)
test_input = tf.random.uniform((1, 10, 512))  # (batch, seq_len, d_model)
test_output = pe(test_input)
print(test_output.shape)  # Should output (1, 10, 512)
print(test_output.dtype)  # Should output float32

# Hyperparameters (from original paper)
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
DROPOUT = 0.1

# Encoder
def encoder(vocab_size):
    inputs = tf.keras.Input(shape=(None,))
    embeddings = tf.keras.layers.Embedding(vocab_size, D_MODEL)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(D_MODEL, tf.float32))
    embeddings = PositionalEncoding(D_MODEL)(embeddings)
    outputs = tf.keras.layers.Dropout(DROPOUT)(embeddings)
    
    for _ in range(NUM_LAYERS):
        attn = tf.keras.layers.MultiHeadAttention(NUM_HEADS, D_MODEL//NUM_HEADS)
        attn_out = attn(outputs, outputs, attention_mask=create_padding_mask(inputs))
        attn_out = tf.keras.layers.Dropout(DROPOUT)(attn_out)
        outputs = tf.keras.layers.LayerNormalization()(outputs + attn_out)
        
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(D_FF, activation='relu'),
            tf.keras.layers.Dense(D_MODEL)
        ])
        ffn_out = ffn(outputs)
        outputs = tf.keras.layers.LayerNormalization()(outputs + ffn_out)
        
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Decoder
def decoder(vocab_size):
    inputs = tf.keras.Input(shape=(None,))
    encoder_outputs = tf.keras.Input(shape=(None, D_MODEL))
    
    embeddings = tf.keras.layers.Embedding(vocab_size, D_MODEL)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(D_MODEL, tf.float32))
    embeddings = PositionalEncoding(D_MODEL)(embeddings)
    outputs = tf.keras.layers.Dropout(DROPOUT)(embeddings)
    
    for _ in range(NUM_LAYERS):
        attn = tf.keras.layers.MultiHeadAttention(NUM_HEADS, D_MODEL//NUM_HEADS)
        attn_out = attn(outputs, outputs, attention_mask=create_look_ahead_mask(tf.shape(inputs)[1]))
        attn_out = tf.keras.layers.Dropout(DROPOUT)(attn_out)
        outputs = tf.keras.layers.LayerNormalization()(outputs + attn_out)
        
        cross_attn = tf.keras.layers.MultiHeadAttention(NUM_HEADS, D_MODEL//NUM_HEADS)
        cross_attn_out = cross_attn(outputs, encoder_outputs, attention_mask=create_padding_mask(inputs))
        cross_attn_out = tf.keras.layers.Dropout(DROPOUT)(cross_attn_out)
        outputs = tf.keras.layers.LayerNormalization()(outputs + cross_attn_out)
        
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(D_FF, activation='relu'),
            tf.keras.layers.Dense(D_MODEL)
        ])
        ffn_out = ffn(outputs)
        outputs = tf.keras.layers.LayerNormalization()(outputs + ffn_out)
        
    return tf.keras.Model(inputs=[inputs, encoder_outputs], outputs=outputs)

# Transformer
def transformer():
    encoder_inputs = tf.keras.Input(shape=(None,))
    decoder_inputs = tf.keras.Input(shape=(None,))
    
    encoder_outputs = encoder(tokenizer_de.vocab_size + 2)(encoder_inputs)  # +2 for START/END tokens
    decoder_outputs = decoder(tokenizer_en.vocab_size + 2)([decoder_inputs, encoder_outputs])
    
    outputs = tf.keras.layers.Dense(tokenizer_en.vocab_size + 2)(decoder_outputs)
    return tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

# Build the model
model = transformer()


# Custom learning rate scheduler
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Compile
learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


EPOCHS = 20
model.fit(dataset, epochs=EPOCHS)


def translate(sentence):
    # Preprocess input
    encoded = START_TOKEN + tokenizer_de.encode(sentence) + END_TOKEN
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        [encoded], maxlen=max_len_de, padding='post', truncating='post'
    )
    
    # Initialize decoder input
    decoder_input = tf.expand_dims(START_TOKEN, 0)
    
    for _ in range(max_len_en):
        predictions = model.predict((padded, decoder_input), verbose=0)
        predicted_id = tf.argmax(predictions[0, -1, :]).numpy()
        
        if predicted_id == END_TOKEN[0]:
            break
            
        decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=-1)
    
    return tokenizer_en.decode(decoder_input[0].numpy().tolist()[1:])  # Remove START token
