import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import random
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_arg():
    """
    argument parser
    :return: arguments
    """
    parser = argparse.ArgumentParser(description='Neural network used to predict pneumonia from lungs radiographies')
    parser.add_argument('data', nargs=3, metavar='data', type=str, help="The datas to be used (train set, test set and"
                                                                        " validation set)")
    return parser.parse_args()


def modelp():
    """
    CNN model with 4 conv layer
    :return: a model
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation="softmax"))
    return model


def modelt():
    """
    a transfer model, not functional
    :return:
    """
    shape = (224, 224, 3)
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=layers.Input(shape=shape))
    for layer in model.layers:
        layer.trainable = False
    model = models.Model(inputs=model.inputs, outputs=model.output[-2])
    # mod_input = layers.Input(shape=shape)
    # beg = model(mod_input)
    mod_input = model.output
    flat1 = layers.Flatten()(mod_input)
    dp1 = layers.Dropout(0.5)(flat1)
    den1 = layers.Dense(64, activation="relu")(dp1)
    dp2 = layers.Dropout(0.5)(den1)
    output = layers.Dense(2, activation="softmax")(dp2)
    t_model = models.Model(model.input, output)
    return t_model

def modlvgg():
    shape = (224, 224, 3)
    vgg= VGG16(weights='imagenet', include_top=False,input_tensor=layers.Input(shape=shape), classes = 2)
    # Freezer les couches du VGG16
    for layer in vgg.layers:
        layer.trainable = False
    out = vgg.output
    x = layers.Flatten()(out)
    x = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=vgg.input, outputs=x)
    model.summary()
    return model


def get_data(path, size):
    """
    retrieves datas to be used to train and test the model and turn the image into an array
    :param path: global path to the data
    :param size: size to resize the images
    :return: an array of data
    """
    data = []
    labels = ["PNEUMONIA", "NORMAL"]
    for label in labels:
        fullpath = os.path.join(path, label)
        class_num = labels.index(label)
        for img in os.listdir(fullpath):
            arr_img = cv2.imread(os.path.join(fullpath, img), cv2.IMREAD_GRAYSCALE)
            # arr_col = cv2.cvtColor(arr_img, cv2.COLOR_GRAY2RGB)
            arr_res = cv2.resize(arr_img, (size, size))
            data.append([arr_res, class_num])
            random.shuffle(data)
    return np.array(data)


def preprocessing_data(data):
    """
    preprocess the data by splitting the images and labels in two different arrays
    :param data: array of images
    :return: two arrays one of images and one of labels
    """
    print("   Preprocessing ongoing...")
    x_data = []
    y_data = []
    for img, label in data:
        x_data.append(img)
        y_data.append(label)
    return x_data, y_data


def data_normalization(xdata, ydata, size):
    """
    Normalizes the images and turn the arrays into numpy arrays
    :param xdata: an array of images
    :param ydata: an array of labels
    :param size: size of the images
    :return: two numpy arrays (images, labels
    """
    print("   Normalization ongoing")
    xdata = np.array(xdata) / 255
    xdata = xdata.reshape(-1, size, size, 1)
    return xdata, np.array(ydata)


def cut_list(data, cut):
    """
    Cuts a given list into two
    :param data: list to be cut
    :param cut: index where to cut
    :return: two lists
    """
    train = data[:cut]
    val = data[cut:]
    return train, val


def dataview(data):
    """
    Plots distibution of data
    :param data: list of labels
    :return: a plot 
    """
    count = []
    for value in data:
        if value == 0:
            count.append("PNEUMONIA")
        else:
            count.append("NORMAL")
    sns.countplot(count)
    plt.show()


def main():
    print("Program launched")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    args = get_arg()
    print("Retrieving data")
    set_train = get_data(args.data[0], 224)
    print("   training data retrieved")
    set_test = get_data(args.data[1], 224)
    print("   testing data retrieved")
    set_val = get_data(args.data[2], 224)
    print("   validation data retrieved")
    print("Preprocessing starting...")
    xtrain, ytrain = preprocessing_data(set_train)
    print("   training preprocessing over")
    xtest, ytest = preprocessing_data(set_test)
    print("   testing preprocessing over")
    xval, yval = preprocessing_data(set_val)
    print("   validation preprocessing over")
    dataview(ytrain)
    dataview(yval)
    dataview(ytest)
    print("Retrieving training and validation data")
    fulldatx = xtrain + xval
    fulldaty = ytrain + yval
    cut = int(len(fulldatx) * 0.9)
    xtrain, xval = cut_list(fulldatx, cut)
    ytrain, yval = cut_list(fulldaty, cut)
    print("retrieving done")
    print("Normalisation starting...")
    x_train, y_train = data_normalization(xtrain, ytrain, 224)
    print("   training normalization over")
    x_test, y_test = data_normalization(xtest, ytest, 224)
    print("   testing normalization over")
    x_val, y_val = data_normalization(xval, yval, 224)
    data_augmented = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
                                        , shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    print("Writing model...")
    model1 = modelp()
    model1.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("Model learning...")
    history1 = model1.fit(data_augmented.flow(x_train, y_train, batch_size=128), batch_size=128, epochs=15,
                          validation_data=data_augmented.flow(x_val, y_val))
    model1.summary()
    print(history1.history.keys())
    print("Model evaluation:")
    eval_model = model1.evaluate(x_test, y_test)
    print(f"Model loss : {eval_model[0]}")
    print(f"Model accuracy : {eval_model[1] * 100}%")
    # summarize history for loss
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for accuracy
    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    print("Program done")


if __name__ == '__main__':
    main()
