from keras import Input, Model
from keras.layers import Dense


def baseline():
    inputs = Input(shape=(4,))
    predictions = Dense(1)(inputs)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def small():
    inputs = Input(shape=(4,))
    x = Dense(32, activation='linear')(inputs)
    predictions = Dense(1)(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def large():
    inputs = Input(shape=(4,))
    x = Dense(32, activation='linear')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='linear')(x)
    predictions = Dense(1)(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
