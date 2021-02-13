import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.regularizers import l2
import matplotlib.pyplot as plt
from dataset_creator import split_dataset
import numpy as np


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


def show_winnings(predicted_classes, actual_classes, odds):
    winnings = 0.0
    for i in range(predicted_classes.shape[0]):
        if predicted_classes[i] == actual_classes[i]:
            winnings = winnings + odds[i][actual_classes[i]] - 1.0
        else:
            winnings = winnings - 1.0
    print(str(winnings))


def show_accuracy_for_classes(predicted_classes, actual_classes):
    predicted_classes_as_int = predicted_classes
        # np.argmax(predicted_classes, axis=1)
    actual_classes_as_int = actual_classes
        # np.argmax(actual_classes, axis=1)
    comparison_array = actual_classes_as_int == predicted_classes_as_int
    for i in np.unique(actual_classes_as_int):
        current_actual_class_indexes = [index for index, class_value in enumerate(actual_classes_as_int) if class_value == i]
        current_predicted_class_indexes = [index for index, class_value in enumerate(predicted_classes_as_int) if class_value == i]
        true_positives = sum(1 for comparison in comparison_array[current_actual_class_indexes] if comparison)
        false_positives = sum(1 for comparison in comparison_array[current_predicted_class_indexes] if not comparison)
        all_actual_class_examples = len(current_actual_class_indexes)
        all_predicted_class_examples = len(current_predicted_class_indexes)
        print("Skutecznosc dla klasy " + str(i) + " = " + str(true_positives/all_actual_class_examples) + "% (" +
              str(true_positives) + "/" + str(all_actual_class_examples) + ")")
        print("Ilosc falszywie przewidzianych dla klasy " + str(i) + " = " + str(false_positives / all_predicted_class_examples)
              + "% (" + str(false_positives) + "/" + str(all_predicted_class_examples) + ")")


def perform_nn_learning(dataset):
    x, y, odds = split_dataset(dataset)
    factor = 1e-5
    rate = 0.1

    # layer = keras.Normalization()
    # layer.adapt(x)
    model_input = keras.Input(shape=(x.shape[1],))
    # model = layer(model_input)
    model = keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(factor))(model_input)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(factor))(model)
    model = keras.layers.Dropout(rate)(model)
    # model = keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(factor))(model)
    # model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(factor))(model)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(factor))(model)
    model = keras.layers.Dropout(rate)(model)
    # model = keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(factor))(model)
    # model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(factor))(model)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(factor))(model)
    model = keras.layers.Dropout(rate)(model)
    output = keras.layers.Dense(3, activation='softmax')(model)
    model = keras.Model(inputs=model_input, outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x, y, epochs=100, batch_size=64, verbose=2, shuffle=True, validation_split=0.15)

    y_prob = model.predict(x)
    y_classes = y_prob.argmax(axis=-1)
    show_winnings(y_classes, y.argmax(axis=-1), odds)
    show_accuracy_for_classes(y_classes, y.argmax(axis=-1))

    plot_metric(history, 'accuracy')
