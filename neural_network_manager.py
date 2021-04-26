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
        # todo: czytelniej
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
                                  win_away * (odds_b - 1) + (1 - win_away) * -1,
                                  tf.zeros_like(odds_a)], axis=1)
    zerod_prediction = tf.where(
        tf.not_equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred),
        tf.zeros_like(y_pred),
        tf.ones_like(y_pred)
    )
    return tf.reduce_mean(tf.reduce_sum(gain_loss_vector * zerod_prediction, axis=1))


def how_many_no_bets(y_true, y_pred):
    all_predictions = y_pred[:, 0:4]
    classes = tf.math.argmax(all_predictions, 1)
    wanted_class = tf.constant(3, dtype="int64")
    logical = tf.math.equal(classes, wanted_class)
    return tf.reduce_sum(tf.cast(logical, tf.float32)) * 100.0 / tf.cast(tf.shape(y_pred)[0], tf.float32)


def create_NN_model(x_train):
    factor = 0.00065
    rate = 0.05

    # tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(4096, activation='relu',
                                 activity_regularizer=l2(factor),
                                 kernel_regularizer=l2(factor), kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, activation='relu',
                                 # activity_regularizer=l2(factor),
                                 kernel_regularizer=l2(factor), kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.Dropout(rate))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(512, activation='relu',
    #                              # activity_regularizer=l2(factor / 2),
    #                              kernel_regularizer=l2(factor), kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor), kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.Dropout(rate))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.Dropout(rate / 2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor/10),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.Dropout(rate / 2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(32, activation='relu',
                                 # activity_regularizer=l2(factor / 2),
                                 kernel_regularizer=l2(factor/10),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    # model.add(keras.layers.Dropout(rate / 4))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(16, activation='relu',
                                 # activity_regularizer=l2(factor / 4),
                                 kernel_regularizer=l2(factor/10),
                                 kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(keras.layers.Dense(4, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal()))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=odds_loss,
                  optimizer=opt,
                  metrics=[how_many_no_bets, only_best_prob_odds_profit])
    return model


def save_model(model):
    model.save(saved_model_location, overwrite=True)


def load_model():
    return keras.models.load_model(saved_model_location)


def perform_nn_learning(model, train_set, val_set):
    x_train = train_set[0]
    y_train = train_set[1]

    # tf.compat.v1.disable_eager_execution()
    history = model.fit(x_train, y_train, epochs=400, batch_size=128, verbose=1, shuffle=False, validation_data=val_set[0:2],
                        callbacks=[EarlyStopping(patience=75, monitor='val_only_best_prob_odds_profit', mode='max', verbose=1),
                                   ModelCheckpoint(saved_weights_location, save_best_only=True, save_weights_only=True, monitor='val_only_best_prob_odds_profit',
                                                   mode='max', verbose=1)]
                                   # TensorBoard(write_grads=True, histogram_freq=1, log_dir='.\\tf_logs', write_images=True, write_graph=True)]
                        )

    model.load_weights(saved_weights_location)

    # TODO: wydzielic do funkcji
    print("Treningowy zbior: ")
    y_train_prob = model.predict(x_train)
    y_train_classes = y_train_prob.argmax(axis=-1)
    train_set_y = y_train[:, 0:4]
    train_bets = y_train[:, 4:7]
    show_winnings(y_train_classes, train_set_y.argmax(axis=-1), train_bets)
    show_accuracy_for_classes(y_train_classes, train_set_y.argmax(axis=-1))

    print("Walidacyjny zbior: ")
    y_prob = model.predict(val_set[0])
    y_classes = y_prob.argmax(axis=-1)
    val_set_y = val_set[1][:, 0:4]
    bets = val_set[1][:, 4:7]
    show_winnings(y_classes, val_set_y.argmax(axis=-1), bets)
    show_accuracy_for_classes(y_classes, val_set_y.argmax(axis=-1))

    plot_metric(history, 'loss')
    plot_metric(history, 'only_best_prob_odds_profit')
    save_model(model)
    return model
