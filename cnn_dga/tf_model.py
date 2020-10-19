import tensorflow as tf
import numpy as np
from IPython import embed
import data


def define_model(max_features, maxlen):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(maxlen,)),
        tf.keras.layers.Embedding(max_features, 100, input_length=maxlen),
        tf.keras.layers.Conv1D(256, 4, 1, activation='relu'),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.005),
        metrics=['accuracy']
    )

    return model


def main():
    print("fetching data...")
    indata = data.get_data()
    domains, labels = zip(*indata)
    char2ix = {x: idx+1 for idx, x in enumerate(set(''.join(domains)))}
    ix2char = {ix: char for char, ix in char2ix.items()}

    # Convert characters to int and pad
    encoded_domains = [[char2ix[y] for y in x] for x in domains]
    encoded_labels = [0 if x == 'benign' else 1 for x in labels]

    max_features = len(char2ix) + 1
    maxlen = np.max([len(x) for x in encoded_domains])

    encoded_labels = np.asarray([
        label
        for idx, label in enumerate(encoded_labels)
        if len(encoded_domains[idx]) > 1
    ])

    encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]
    assert len(encoded_domains) == len(encoded_labels)
    padded_domains = tf.keras.preprocessing.sequence.pad_sequences(encoded_domains, maxlen)
    
    train_ds, test_ds = data.prepare_data(padded_domains, encoded_labels)

    model = define_model(max_features, maxlen)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(
        train_ds, epochs=10,
        validation_data=test_ds,
        validation_steps=30,
        callbacks=[callback]
    )

    embed()

if __name__ == "__main__":
    main()
