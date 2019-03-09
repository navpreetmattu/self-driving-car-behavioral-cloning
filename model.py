import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Flatten, Lambda, Conv2D, Dropout, Cropping2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Reading the csv file
def read_data(path):
    ''' This function reads the csv log file containing all the info related to 
        image data and steering angles.
    '''
    samples=[]
    with open(path) as file:
        reader = csv.reader(file)
        next(reader) # Since first row contains headers
        for line in reader:
            samples.append(line)
    return samples
    

def data_generator(data, is_udacity_data, batch_size=32):
    ''' This generator function is used to read the image data and to create batches of it
        to feed the model for training and validation purpose, one batch at a time.
    '''
    data_size = len(data)
    # As the image path in data provided by udacity starts from IMG folder, 
    # concatinating the remaining path to it. 
    images_path = '/opt/carnd_p3/data/' if is_udacity_data else ''
    while True:
        shuffle(data)
        for offset in range(0, data_size, batch_size):
            batch_samples = data[offset:offset+batch_size]
            images=[]
            angles=[]
            for line in batch_samples:
                image_center = cv2.cvtColor(cv2.imread(images_path + line[0].strip()), cv2.COLOR_BGR2RGB)
                image_left = cv2.cvtColor(cv2.imread(images_path + line[1].strip()), cv2.COLOR_BGR2RGB)
                image_right = cv2.cvtColor(cv2.imread(images_path + line[2].strip()), cv2.COLOR_BGR2RGB)
                images.extend([image_center, image_left, image_right])

                correction_factor = .2
                angle_center = float(line[3])
                angle_left = angle_center + correction_factor
                angle_right = angle_center - correction_factor
                angles.extend([angle_center, angle_left, angle_right])

                # Augmented the data by flipping the images horizontally and changing the angle sign
                images.extend([cv2.flip(image_center, 1), cv2.flip(image_left, 1), cv2.flip(image_right, 1)])
                angles.extend([angle_center*-1., angle_left*-1., angle_right*-1.])

            X_train = np.asarray(images)
            y_train = np.asarray(angles)
            yield shuffle(X_train, y_train)


def nvidia_model():
    ''' This function creates a convolutional neural network with NVIDIA model 
        architecture. It uses Adam optimizer and Mean Square Error as loss function.
        It uses ELU as activation function and uses dropout to avoid overfitting.
    '''
    model = Sequential()   
    model.add(Cropping2D(((70, 20), (0, 0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.) - .5))
    model.add(Conv2D(24, (5, 5), strides=2, activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=2, activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=2, activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.4))
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dropout(0.4))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


def train_model(model, samples, model_name, epochs, test_data_size, batch_data_size, is_udacity_data):
    ''' This function will split the data into train and validation sets, 
        create train and validation generators and train the model for number of epochs passed. 
        It saves the model after training.
    '''
    # Splitting the data into 80% training and 20% validation set
    train_samples, validation_samples = train_test_split(samples, test_size=test_data_size)
    train_generator = data_generator(train_samples, is_udacity_data, batch_size=batch_data_size)
    validation_generator = data_generator(validation_samples, is_udacity_data, batch_size=batch_data_size)

    model.fit_generator(train_generator, steps_per_epoch=len(train_samples), 
                        validation_data=validation_generator, 
                        validation_steps=len(validation_samples), epochs=epochs, verbose=1)

    # Saving the model
    model.save(model_name)
    

model_name = 'model.h5'

# Fetching udacity provided data
samples = read_data('/opt/carnd_p3/data/driving_log.csv')
# Creating model object
model = nvidia_model()
# Training the model with train data for 5 epochs with batch size of 32
train_model(model, samples, model_name, epochs=5, test_data_size=.2, batch_data_size=32, is_udacity_data=True)

# Fetching data created by myself
samples = read_data('/opt/train_data/driving_log.csv')
# Reading saved model
model = load_model(model_name)
# Training the model with train data for 5 epochs with batch size of 32
train_model(model, samples, model_name, epochs=5, test_data_size=.2, batch_data_size=32, is_udacity_data=False)

print(model.summary())