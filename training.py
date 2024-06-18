import numpy as np # linear algebra
import pandas as pd  # data processing
import os #  to interact with files using there paths
from sklearn.datasets import load_files
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def process(path):
    
    data_dir =path
     
    #Loading our Data
    data = load_files(data_dir)
    X = np.array(data['filenames'])
    y = np.array(data['target'])
    labels = np.array(data['target_names'])
    def convert_img_to_arr(file_path_list):
        arr = []
        #size=64,64
        img_width, img_height = 150,150
        for file_path in file_path_list:
            img = load_img(file_path, target_size = (img_width, img_height))
            img = img_to_array(img)
            arr.append(img)
            #arr.append(cv2.resize(img,size))
        return arr
     
    X = np.array(convert_img_to_arr(X))
    print(X.shape)

    # Let's resize or rescale training data
    X = X.astype('float32')/255
     
    # Let's confirm the number of classes :) 
    no_of_classes = len(np.unique(y))
    print("no_of_classes==",no_of_classes)
    y = np.array(np_utils.to_categorical(y,no_of_classes))


    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    print('The test Data Shape ', X_test.shape[0])
     
    X_test, X_valid, y_test, y_valid = train_test_split(X_test,y_test, test_size = 0.5)
    print('The training Data Shape ', X_valid.shape[0])
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=X_train.shape[1:], activation='relu', name='Conv2D_1'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', name='Conv2D_2'))
    model.add(MaxPool2D(pool_size=(2,2), name='Maxpool_1'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_3'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='Conv2D_4'))
    model.add(MaxPool2D(pool_size=(2,2), name='Maxpool_2'))
    model.add(Dropout(0.25))
        
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_5'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', name='Conv2D_6'))
    model.add(MaxPool2D(pool_size=(2,2), name='Maxpool_3'))

    model.add(Flatten())
    model.add(Dense(units=512, activation='relu', name='Dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation='relu', name='Dense_2'))
    model.add(Dense(units=no_of_classes, activation='softmax', name='Output'))
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = 100
    batch_size=32
     
    train_datagen = ImageDataGenerator(
            rotation_range=10,  
            zoom_range = 0.1, 
            width_shift_range=0.1,
            height_shift_range=0.1,  
            horizontal_flip=True)
     
    test_datagen = ImageDataGenerator()
     
    train_generator = train_datagen.flow(
        X_train,y_train,
        batch_size=batch_size)
     
    validation_generator = test_datagen.flow(
        X_valid,y_valid,
        batch_size=batch_size)
     
    checkpointer = ModelCheckpoint(filepath = "PId_Best.h5", save_best_only = True, verbose = 1)
    learning_rate_reduction=ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose = 1, factor = 0.5, minlr = 0.00001)
     

    start = time.time()
     
    # let's get started !
     
    history=model.fit_generator(train_generator,
                                epochs=epochs,
                                validation_data = validation_generator,
                                verbose=1,
                                steps_per_epoch=len(X_train) // batch_size,
                                validation_steps=len(X_valid) //batch_size,
                                callbacks=[checkpointer, learning_rate_reduction])
     
    end = time.time()
     
    duration = end - start
    print ('\n This Model took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, epochs) )


    def plot(history):
        plt.figure(1)
        #plt.figure(figsize=(10,10)) 
     
         # summarize history for accuracy  
     
        plt.subplot(211)  
        plt.plot(history.history['acc'])  
        plt.plot(history.history['val_acc'])  
        plt.title('accuracy vs val_accuracy')  
        plt.ylabel('accuracy')  
        plt.xlabel('epoch')  
        plt.legend(['Train', 'Validation'], loc='lower right')  
     
         # summarize history for loss  
     
        plt.subplot(212)  
        plt.plot(history.history['loss'])  
        plt.plot(history.history['val_loss'])  
        plt.title('loss vs val_loss')  
        plt.ylabel('loss')  
        plt.xlabel('epoch')  
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.tight_layout()
        #plt.save("plot.png")
        plt.show()
        
        
     
    # Finaly, let's call the plot function with the 'result' parameter 
     
    plot(history)
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred,axis = 1)
    Y_true = np.argmax(y_test,axis = 1)
    confusion_mtx = confusion_matrix(Y_true,Y_pred_classes)
    f,ax = plt.subplots(figsize = (8,8))
    sns.heatmap(confusion_mtx,annot=True,linewidths = 0.01,cmap="Greens",
                linecolor = "gray",fmt = ".2f",ax=ax
                )
    plt.xlabel("predicted label")
    plt.ylabel("True Label")
    plt.title("confusion matrix")
    plt.show()
    plt.savefig("CM_plot.png")
process("./Dataset")