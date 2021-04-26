from enum import Enum
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras.regularizers import l2
import matplotlib.pyplot as plt
from dataset_creator import split_dataset
import numpy as np
import math


class Categories(Enum):
    HOME_WIN = 0
    TIE = 1
    AWAY_WIN = 2
    NO_BET = 3


results_to_description_dict = {0: 'Wygrana gospodarzy', 1: 'Remis', 2: 'Wygrana gości', 3: 'Brak zakładu'}
saved_model_location = "./NN_full_model/"
saved_weights_location = "./NN_model_weights/checkpoint_weights"
confidence_threshold = 0.015


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
    if metric == 'loss':
        # concated_metrics = np.concatenate((np.asarray(train_metrics), np.asarray(val_metrics)))
        # concated_metrics = concated_metrics[concated_metrics < 30]
        # avg = np.average(concated_metrics)
        # std_dev = math.sqrt(np.sum(concated_metrics * concated_metrics) / len(concated_metrics) - avg ** 2)
        # start = avg - 1.25 * std_dev
        # end = avg + 1.25 * std_dev
        # plt.ylim([start, end])
        plt.ylim([0.5, 2])
    plt.show()


def show_winnings(predicted_classes, actual_classes, odds):
    winnings = 0.0
    for i in range(predicted_classes.shape[0]):
        # Jesli siec zdecydowala sie nie obstawiac meczu
        # todo: czytelniej
        if predicted_classes[i] == Categories.NO_BET.value:
            continue
        elif predicted_classes[i] == actual_classes[i]:
            winnings = winnings + odds[i][actual_classes[i]] - 1.0
        else:
            winnings = winnings - 1.0
    print("Bilans wygranych/strat z potencjalnych zakładów: " + str(winnings))


def show_winnings_within_threshold(classes_possibilities, actual_classes, odds):
    winnings = 0.0
    no_bet = 0
    outcome_possibilities = 1.0 / odds
    prediction_diff = classes_possibilities - outcome_possibilities
    chosen_class = classes_possibilities.argmax(axis=-1)
    for i in range(prediction_diff.shape[0]):
        # Jesli siec zdecydowala sie nie obstawiac meczu
        if prediction_diff[i][chosen_class[i]] < confidence_threshold:
            no_bet += 1
            continue
        elif chosen_class[i] == actual_classes[i]:
            winnings = winnings + odds[i][actual_classes[i]] - 1.0
        else:
            winnings = winnings - 1.0
    print("Bilans wygranych/strat z potencjalnych zakładów: " + str(winnings))
    print("Ilosc nieobstawionych zakładów z powodu zbyt niskiej pewnosci: " + str(no_bet))


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


def show_accuracy_within_threshold(classes_possibilities, actual_classes, odds):
    outcome_possibilities = 1.0 / odds
    prediction_diff = classes_possibilities - outcome_possibilities
    chosen_class = classes_possibilities.argmax(axis=-1)
    for index, c in enumerate(chosen_class):
        if prediction_diff[index, c] < confidence_threshold:
            chosen_class[index] = Categories.NO_BET.value
    show_accuracy_for_classes(chosen_class, actual_classes)


def odds_loss(y_true, y_pred):
    win_home_team = y_true[:, 0:1]
    draw = y_true[:, 1:2]
    win_away = y_true[:, 2:3]
    no_bet = y_true[:, 3:4]
    odds_a = y_true[:, 4:5]
    odds_draw = y_true[:, 5:6]
    odds_b = y_true[:, 6:7]
    gain_loss_vector = tf.concat([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
                                  draw * (odds_draw - 1) + (1 - draw) * -1,
                                  win_away * (odds_b - 1) + (1 - win_away) * -1,
                                  tf.zeros_like(odds_a)], axis=1)
    return -1 * tf.reduce_mean(tf.reduce_sum(gain_loss_vector * y_pred, axis=1))


def only_best_prob_odds_profit(y_true, y_pred):
    win_home_team = y_true[:, 0:1]
    draw = y_true[:, 1:2]
    win_away = y_true[:, 2:3]
    no_bet = y_true[:, 3:4]
    odds_a = y_true[:, 4:5]
    odds_draw = y_true[:, 5:6]
    odds_b = y_true[:, 6:7]
    gain_loss_vector = tf.concat([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
                                  draw * (odds_draw - 1) + (1 - draw) * -1,
                                  win_away * (odds_b - 1) + (1 - win_away) * -1
                                  # tf.zeros_like(odds_a)
                                  ], axis=1)
    outcome_possibilities = 1.0/y_true[:, 4:7]
    zerod_prediction = tf.where(
        tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
        tf.zeros_like(y_pred),
        y_pred
    )
    predictions_above_threshold = tf.where(
        tf.greater_equal(tf.subtract(zerod_prediction, outcome_possibilities), confidence_threshold),
        tf.ones_like(y_pred),
        tf.zeros_like(y_pred)
    )
    return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * predictions_above_threshold, axis=1))


def how_many_no_bets(y_true, y_pred):
    all_predictions = y_pred[:, 0:4]
    classes = tf.math.argmax(all_predictions, 1)
    wanted_class = tf.constant(3, dtype="int64")
    logical = tf.math.equal(classes, wanted_class)
    return tf.reduce_sum(tf.cast(logical, tf.float32)) * 100.0 / tf.cast(tf.shape(y_pred)[0], tf.float32)


def categorical_crossentropy_with_bets(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true[:, 0:3], y_pred)


def categorical_acc_with_bets(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true[:, 0:3], y_pred)


def create_NN_model(x_train):
    factor = 0.0003
    rate = 0.05

    # tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(4096, activation='relu',
                                 activity_regularizer=l2(factor/2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(1024, activation='relu',
    #                              activity_regularizer=l2(factor/2),
    #                              kernel_regularizer=l2(factor),
    #                              kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.Dropout(rate))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1024, activation='relu',
                                 activity_regularizer=l2(factor/2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu',
                                 # activity_regularizer=l2(factor/4),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate / 2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu',
                                 # activity_regularizer=l2(factor / 10),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate / 2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(32, activation='relu',
                                 # activity_regularizer=l2(factor / 10),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dropout(rate / 4))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(16, activation='relu',
                                 # activity_regularizer=l2(factor / 10),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal()))
    model.compile(loss=categorical_crossentropy_with_bets,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0015),
                  metrics=[categorical_acc_with_bets, only_best_prob_odds_profit])
    # only_best_prob_odds_profit
    return model


def save_model(model):
    model.save(saved_model_location, overwrite=True)


def load_model():
    return keras.models.load_model(saved_model_location)


def eval_model_after_learning(y_true, y_pred, odds):
    y_pred_classes = y_pred.argmax(axis=-1)
    y_true_classes = y_true.argmax(axis=-1)
    show_winnings_within_threshold(y_pred, y_true_classes, odds)
    show_accuracy_within_threshold(y_pred, y_true_classes, odds)


def perform_nn_learning(model, train_set, val_set):
    x_train = train_set[0]
    y_train = train_set[1]
    x_val = val_set[0]
    y_val = val_set[1]

    # tf.compat.v1.disable_eager_execution()
    history = model.fit(x_train, y_train, epochs=350, batch_size=128, verbose=1, shuffle=False, validation_data=val_set[0:2],
                        callbacks=[EarlyStopping(patience=60, monitor='val_only_best_prob_odds_profit', mode='max', verbose=1),
                                   ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True,
                                                   monitor='val_only_best_prob_odds_profit',
                                                   mode='max', verbose=1)]
                        # TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_images=True, write_graph=True)]
                        )

    model.load_weights(saved_weights_location)

    print("Treningowy zbior: ")
    eval_model_after_learning(y_train[:, 0:3], model.predict(x_train), y_train[:, 4:7])
    print("Walidacyjny zbior: ")
    eval_model_after_learning(y_val[:, 0:3], model.predict(x_val), y_val[:, 4:7])

    plot_metric(history, 'loss')
    plot_metric(history, 'only_best_prob_odds_profit')
    plot_metric(history, 'categorical_acc_with_bets')
    save_model(model)
    return model
