import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from finacial_sentiment import *

class TextProc(TextProcessing):
    def __init__(self, file):
        self.df = pd.read_csv(file, error_bad_lines=False)
        self.sent = self.df['text'].astype(str).str.lower()  # text not cleaned
        self.labels = self._process_labels()

    def _process_labels(self):
        labels = self.df['verified'].apply(lambda x: 0 if x == False else 1)
        return labels

def define_hud(dropout_rate, n1, n2, num_class, activation_func):
    hub_layer = hub.KerasLayer\
        ("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
         trainable=True, input_shape=[], dtype=tf.string)
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(n1, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    #model.add(tf.keras.layers.Dense(n2, activation='relu'))
    #model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(num_class, activation=activation_func))
    return model

# graph the performance of model
def graph_plots(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

# compute class weights for imbalanced data
def compute_class_weights(labels):
    n_samples = len(labels)
    n_classes = len(labels.unique())
    class_weights = {}
    class_names = labels.value_counts().index.tolist()
    for i in range(len(labels.value_counts())):
        class_weights[class_names[i]] = round(n_samples/\
                                (n_classes * labels.value_counts()[i]), 2)
    return class_weights


