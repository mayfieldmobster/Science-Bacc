import tensorflow.keras as keras


def cifar10_model():
    model = keras.Sequential()
    model.add(keras.applications.ResNet50(include_top=False, weights=None, input_tensor=keras.Input(shape=(32,32,3)), classes=10, pooling='avg' ))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def cifar100_model():
    model = keras.Sequential()
    model.add(keras.applications.ResNet50(include_top=False, weights=None, input_tensor=keras.Input(shape=(32,32,3)), classes=100, pooling='avg' ))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='softmax'))
    return model

def image_net_model():
    model = keras.Sequential()
    model.add(keras.applications.ResNet50(include_top=False, weights=None, input_tensor=keras.Input(shape=(32,32,3)), classes=1000, pooling='avg'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000, activation='softmax'))
    return model

#cifar10_model = cifar10_model()