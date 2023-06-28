import shutil
import pandas as pd
from PIL import Image, ImageFont
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import tensorflow
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from models.models import vgg_16, original, resnet_50v2, inception_v3, densenet_121
from params import IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_NUM, MODIFIED_IMAGES_DIRECTORY, MODIFIED_TEST_IMAGES_DIRECTORY
from side_funcs import plot_model_history, plot_confusion_matrix
import visualkeras
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils.vis_utils import plot_model
import math
from plot_keras_history import show_history, plot_history


EPOCHS_NUM = 50
BATCH_SIZE = 64

models = ['vgg', 'resnet', 'densenet', 'inception', 'original']
model_name = models[4]

optimizers = ['sgd', 'adam']
optimizer_name = optimizers[1]


# Only unique values
df = pd.read_csv('./meta_data/HAM10000_metadata-unique_data.csv')

# df = df[:2000]
image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join('./HAM_RESIZED/', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# TRAINING DATASET
df['path'] = df['image_id'].map(image_path.get)
df['cell_type'] = df['dx'].map(lesion_type_dict.get)
df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
df['age'].fillna((df['age'].mean()), inplace=True)
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x)))

# # ******************************************** RESIZE IMAGES AND SETTING IMAGES WITH DIFF SIZE TO VARIABLE ********************************************
#df_test['image'] = df_test['path'].map(lambda x: np.asarray(Image.open(x).resize((IMAGE_HEIGHT, IMAGE_WIDTH))))
#df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((IMAGE_HEIGHT, IMAGE_WIDTH))))
#test_imagesToSave = df_test['path'].map(lambda x: [os.path.split(x)[1], Image.open(r'{}'.format(x)).resize((IMAGE_HEIGHT, IMAGE_WIDTH))])
#imagesToSave = df['path'].map(lambda x: [os.path.split(x)[1], Image.open(r'{}'.format(x)).resize((IMAGE_HEIGHT, IMAGE_WIDTH))])
# ******************************************** SAVING IMAGES TO NEW DIR, IF IT EXISTS DELETE IT            ********************************************
#dirs = [MODIFIED_IMAGES_DIRECTORY, MODIFIED_TEST_IMAGES_DIRECTORY]
#for dir_name in dirs:
#    dir_path = './{}'.format(dir_name)
#    if os.path.exists(dir_path):
#        shutil.rmtree(dir_path)
#    os.makedirs(dir_path)
#    for image in imagesToSave:
#        image[1].save('./{}/{}'.format(dir_name, image[0]))

features = df.drop(columns=['cell_type_idx'], axis=1)
target = df['cell_type_idx']

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.30, random_state=1234)

print(tensorflow.unique(x_train_o.cell_type.values))
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_test_mean = np.mean(x_test)

x_train_std = np.std(x_train)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std

y_train = to_categorical(y_train_o, num_classes=CLASSES_NUM)
y_test = to_categorical(y_test_o, num_classes=CLASSES_NUM)

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.15, random_state=2)
x_train = x_train.reshape(x_train.shape[0], *(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
x_test = x_test.reshape(x_test.shape[0], *(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

model = None
if model_name == models[0]:
    model = vgg_16()
elif model_name == models[1]:
    model = resnet_50v2()
elif model_name == models[2]:
    model = densenet_121()
elif model_name == models[3]:
    model = inception_v3()
elif model_name == models[4]:
    model = original()



# optymalizatory - SGD, ADAM, SGD ze schedulerem
optimizer = None

if optimizer_name == optimizers[0]:
    optimizer = SGD(learning_rate=0.01, momentum=0.0)
elif optimizer_name == optimizers[1]:
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy')


learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# DATA AUGMENTATION
# Random ----> Rotation: 10 deg, Zoom 10%, Shift horizontally: 10%, Shift vertically: 10%
data_gen = ImageDataGenerator(featurewise_center=False,
                              samplewise_center=False,
                              featurewise_std_normalization=False,
                              samplewise_std_normalization=False,
                              zca_whitening=False,
                              rotation_range=10,
                              zoom_range=0.1,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              horizontal_flip=False,
                              vertical_flip=False)

data_gen.fit(x_train)
data = data_gen.flow(x_train, y_train, batch_size=BATCH_SIZE)
fit_batch_size = int(BATCH_SIZE/4)
history = model.fit(data,
                    epochs=EPOCHS_NUM,
                    batch_size=fit_batch_size,
                    validation_data=(x_validate, y_validate),
                    validation_steps=x_test.shape[0],
                    verbose=1,
                    steps_per_epoch=math.floor(x_train.shape[0]/BATCH_SIZE),
                    callbacks=[learning_rate_reduction])


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
print(model.evaluate(x_test, y_test, verbose=1))
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./results/accuracy_' + model_name + '_' + optimizer_name + '_' + str(EPOCHS_NUM) + '.svg')
# plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./results/loss_' + model_name + '_' + optimizer_name + '_' + str(EPOCHS_NUM) + '.svg')
# plt.show()
plt.close()

# Predict the values from the validation dataset
Y_pred = model.predict(x_validate)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_validate, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

target_names = lesion_type_dict.values()

print(classification_report(y_true=Y_true, y_pred=Y_pred_classes, labels=[0, 1, 2, 3, 4, 5, 6]))

plot_confusion_matrix(confusion_mtx, classes=range(CLASSES_NUM))

label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
plt.bar(np.arange(CLASSES_NUM), label_frac_error)
plt.xlabel('True Label')
plt.ylabel('Fraction classified incorrectly')
