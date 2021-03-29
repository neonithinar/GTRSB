import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from configurations.configs import Configurations

Configs = Configurations()


def SE_block(tensor, ratio = 16):
    """
    Tensorflow Functional API implementation of
    Squeeze and excitation block implementation from https://arxiv.org/pdf/1709.01507
     args:
        tensor: tf.tensor
        ratio: reduction ratio, default value = 16 as suggested by paper for RESNET50
    """
    input_tensor = tensor
    # ch_axis = 1 if keras.backend.image_data_format() == 'channel_first' else -1
    num_filters = getattr(input_tensor, 'shape')[-1]
    se_block_shape = (1, 1, num_filters)

    squeeze = keras.layers.GlobalAveragePooling2D()(input_tensor)
    squeeze = keras.layers.Reshape(se_block_shape)(squeeze)

    excitation = keras.layers.Dense(num_filters // ratio, activation='relu',
                                    kernel_initializer='he_normal', use_bias=False)(squeeze)
    excitation = keras.layers.Dense(num_filters, activation='sigmoid',
                                    kernel_initializer='he_normal', use_bias=False)(excitation)
    return keras.layers.multiply([input_tensor, excitation])

def Build_model():
    """
    Builds the CNN with squeeze and excitation block

    """
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    # input = keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape= (img_height, img_width, 3))
    input = keras.layers.Input(shape=(Configs.img_height, Configs.img_width, 3))
    # print(input.shape)
    rescale = keras.layers.experimental.preprocessing.Rescaling(1. / 255)(input)
    # print(rescale.shape)
    conv1_1 = keras.layers.Conv2D(32, kernel_size=(3, 3),
                                  padding='same', activation='relu')(rescale)
    conv1_2 = keras.layers.Conv2D(32, kernel_size=(3, 3),
                                  padding='same', activation='relu')(conv1_1)
    maxpool_1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1_2)
    se_block_1 = SE_block(maxpool_1)

    conv2_1 = keras.layers.Conv2D(64, kernel_size=(3, 3),
                                  padding='same', activation='relu')(se_block_1)
    conv2_2 = keras.layers.Conv2D(64, kernel_size=(3, 3),
                                  padding='same', activation='relu')(conv2_1)
    maxpool_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2_2)
    BN_2 = keras.layers.BatchNormalization()(maxpool_2)
    dropout_2 = keras.layers.Dropout(0.2)(BN_2)

    conv3_1 = keras.layers.Conv2D(128, kernel_size=(3, 3),
                                  padding='same', activation='relu')(dropout_2)
    conv3_2 = keras.layers.Conv2D(128, kernel_size=(3, 3),
                                  padding='same', activation='relu')(conv3_1)
    maxpool_3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3_2)
    BN_3 = keras.layers.BatchNormalization()(maxpool_3)
    dropout_3 = keras.layers.Dropout(0.2)(BN_3)
    se_block_2 = SE_block(dropout_3)

    Flatten = keras.layers.Flatten()(se_block_2)
    Dense_1 = keras.layers.Dense(128, activation='relu')(Flatten)
    dropout_4 = keras.layers.Dropout(0.5)(Dense_1)
    Dense_2 = keras.layers.Dense(86, activation='relu')(dropout_4)
    BN_4 = keras.layers.BatchNormalization()(Dense_2)
    dropout_5 = keras.layers.Dropout(0.5)(BN_4)
    final_dense = keras.layers.Dense(43, name='top_dense')(dropout_5)
    output = keras.layers.Softmax()(final_dense)

    return keras.models.Model(inputs=[input], outputs=[output])



def Train(train_ds, val_ds):
    """
    Trains a CNN model with squeeze and excitation blocks and saves it
    args:
        summary (Boolean): if True, prints model summary
        plot_model (Boolean): if True display model diagram

    """

    print("training the model")

    model = Build_model()
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, callbacks=[early_stopping_cb], epochs=50)
    model.save('models/CNN_SE_model.h5')
    print('model saved')
    history_df = pd.DataFrame(history.history)
    hist_csv_file = 'models/history.csv'
    with open(hist_csv_file, mode='w') as f:
        history_df.to_csv(f)
    print('history saved')
    return print("model trained successfully")


