import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_mlp(input_dim, n_classes=3, hidden=[128, 64], dropout=0.4, l2=1e-4):
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for h in hidden:
        x = layers.Dense(h, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="MLP")
    return model

# -------------------------------------------------------------------
# 🧠 LSTM Model (lightweight for EEG)
# -------------------------------------------------------------------
def build_lstm(input_shape, n_classes=5, dropout=0.5):
    """
    Lightweight LSTM model for EEG classification
    input_shape: (timesteps, channels)
    """
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="LSTM")
    return model


# -------------------------------------------------------------------
# 🔁 BiLSTM Model (captures past + future temporal context)
# -------------------------------------------------------------------
def build_bilstm(input_shape, n_classes=5, dropout=0.5):
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="BiLSTM")
    return model


# -------------------------------------------------------------------
# ⚙️ 1D CNN Model (extracts local temporal-frequency features)
# -------------------------------------------------------------------
def build_cnn1d(input_shape, n_classes=5, dropout=0.3):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="CNN1D")
    return model


# -------------------------------------------------------------------
# 🧩 Transformer Encoder Model (advanced)
# -------------------------------------------------------------------
def build_transformer_encoder(input_shape, n_classes=5, num_heads=4, ff_dim=128, dropout=0.2):
    """
    Simple Transformer encoder for EEG sequence classification.
    input_shape: (timesteps, channels)
    """
    inp = layers.Input(shape=input_shape)
    model_dim = 64

    # Project EEG channels into model dimension
    x = layers.Dense(model_dim)(inp)

    # Positional encoding (learned embeddings)
    seq_len = input_shape[0]
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=model_dim)
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_encoding = pos_emb(positions)
    x = x + pos_encoding

    # Transformer encoder layers
    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=model_dim // num_heads)(x, x)
        attn = layers.Dropout(dropout)(attn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ff = layers.Dense(ff_dim, activation='relu')(x)
        ff = layers.Dense(model_dim)(ff)
        ff = layers.Dropout(dropout)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="Transformer")
    return model
