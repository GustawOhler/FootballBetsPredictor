from enum import Enum
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
import matplotlib.pyplot as plt
from dataset_creator import split_dataset
import numpy as np


class Categories(Enum):
    HOME_WIN = 0
    TIE = 1
    AWAY_WIN = 2
    NO_BET = 3


results_to_description_dict = {0: 'Wygrana gospodarzy', 1: 'Remis', 2: 'Wygrana gości', 3: 'Brak zakładu'}
saved_model_location = "./NN_full_model/"
saved_weights_location = "./NN_model_weights/checkpoint_weights"


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
        # Jesli siec zdecydowala sie nie obstawiac meczu
        if predicted_classes[i] == Categories.NO_BET.value:
            continue
        elif predicted_classes[i] == actual_classes[i]:
            winnings = winnings + odds[i][actual_classes[i]] - 1.0
        else:
            winnings = winnings - 1.0
    print("Bilans wygranych/strat z potencjalnych zakładów w zbiorze walidacyjnym: " + str(winnings))


def show_accuracy_for_classes(predicted_classes, actual_classes):
    predicted_classes_as_int = predicted_classes
    actual_classes_as_int = actual_classes
    comparison_array = actual_classes_as_int == predicted_classes_as_int
    for i in np.unique(actual_classes_as_int):
        current_actual_class_indexes = [index for index, class_value in enumerate(actual_classes_as_int) if class_value == i]
        current_predicted_class_indexes = [index for index, class_value in enumerate(predicted_classes_as_int) if class_value == i]
        true_positives = sum(1 for comparison in comparison_array[current_actual_class_indexes] if comparison)
        false_positives = sum(1 for comparison in comparison_array[current_predicted_class_indexes] if not comparison)
        all_actual_class_examples = len(current_actual_class_indexes)
        all_predicted_class_examples = len(current_predicted_class_indexes)
        print("Procent odgadniętych przykładów na wszystkie przykłady z klasą \"" + results_to_description_dict[i]
              + "\" = {:.1f}".format(100 * true_positives / all_actual_class_examples if all_actual_class_examples != 0 else 0)
              + "% (" + str(true_positives) + "/" + str(all_actual_class_examples) + ")")
        print("Ilosc falszywie przewidzianych dla klasy \"" + results_to_description_dict[i]
              + "\" = {:.1f}".format(100 * false_positives / all_predicted_class_examples if all_predicted_class_examples != 0 else 0)
              + "% (" + str(false_positives) + "/" + str(all_predicted_class_examples) + ")")
    not_bet_logical = predicted_classes_as_int == Categories.NO_BET.value
    not_bet_sum = sum(1 for logic_1 in not_bet_logical if logic_1)
    all_classes_len = len(predicted_classes_as_int)
    print("Ilosc nieobstawionych zakladow = {:.1f}".format(100 * not_bet_sum / all_classes_len if all_actual_class_examples != 0 else 0)
          + "% (" + str(not_bet_sum) + "/" + str(all_classes_len) + ")")


def odds_loss(y_true, y_pred):
    win_home_team = y_true[:, 0:1]
    draw = y_true[:, 1:2]
    win_away = y_true[:, 2:3]
    no_bet = y_true[:, 3:4]
    odds_a = y_true[:, 4:5]
    odds_draw = y_true[:, 5:6]
    odds_b = y_true[:, 6:7]
    gain_loss_vector = tf.keras.backend.concatenate([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
                                                     draw * (odds_draw - 1) + (1 - draw) * -1,
                                                     win_away * (odds_b - 1) + (1 - win_away) * -1,
                                                     tf.keras.backend.ones_like(odds_a) * -0.03], axis=1)
    return -1 * tf.keras.backend.mean(tf.keras.backend.sum(gain_loss_vector * y_pred, axis=1))


def how_many_no_bets(y_true, y_pred):
    all_predictions = y_pred[:, 0:4]
    classes = tf.math.argmax(all_predictions, 1)
    wanted_class = tf.constant(3, dtype="int64")
    logical = tf.math.equal(classes, wanted_class)
    return tf.reduce_sum(tf.cast(logical, tf.float32)) * 100.0 / tf.cast(tf.shape(y_pred)[0], tf.float32)


def create_keras_model(x_train):
    factor = 0.001
    rate = 0.1

    model_input = keras.Input(shape=(x_train.shape[1],))
    model = keras.layers.BatchNormalization()(model_input)
    model = keras.layers.Dense(2048, activation='relu',
                               activity_regularizer=l2(factor),
                               kernel_regularizer=l2(factor))(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(2048, activation='relu', activity_regularizer=l2(factor),
                               kernel_regularizer=l2(factor))(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(1024, activation='relu', activity_regularizer=l2(factor),
                               kernel_regularizer=l2(factor))(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(512, activation='relu', activity_regularizer=l2(factor),
                               kernel_regularizer=l2(factor))(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(256, activation='relu', activity_regularizer=l2(factor),
                               kernel_regularizer=l2(factor))(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(128, activation='relu', activity_regularizer=l2(factor),
                               kernel_regularizer=l2(factor))(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(64, activation='relu', activity_regularizer=l2(factor),
                               kernel_regularizer=l2(factor))(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(rate)(model)
    model = keras.layers.Dense(32, activation='relu', activity_regularizer=l2(factor),
                               kernel_regularizer=l2(factor))(model)
    output = keras.layers.Dense(3, activation='softmax')(model)
    model = keras.Model(inputs=model_input, outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def save_model(model):
    model.save(saved_model_location, overwrite=True)


def load_model():
    return keras.models.load_model(saved_model_location)


def perform_nn_learning(model, train_set, val_set):
    x_train = train_set[0]
    y_train = train_set[1][:, 0:3]
    x_val = val_set[0]
    y_val = val_set[1][:, 0:3]

    history = model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=1, shuffle=False, validation_data=(x_val, y_val),
                        callbacks=[EarlyStopping(patience=5, verbose=1),
                                   ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True, verbose=1)])

    model.load_weights(saved_weights_location)

    y_prob = model.predict(val_set[0])
    y_classes = y_prob.argmax(axis=-1)
    val_set_y = val_set[1][:, 0:3]
    bets = val_set[1][:, 4:7]
    show_winnings(y_classes, val_set_y.argmax(axis=-1), bets)
    show_accuracy_for_classes(y_classes, val_set_y.argmax(axis=-1))

    plot_metric(history, 'loss')
    save_model(model)
    return model
