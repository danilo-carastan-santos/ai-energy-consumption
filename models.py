# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tensorflow imports
from tensorflow.keras import layers, models, regularizers

# Defining a simple convolutional model
def get_simple_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # display the model's architecture so far
    #model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # display the complete architecture
    model.summary()
    return model

# Model obtained from Schuler, Joao Paulo Schwarz, et al. "Grouped Pointwise
# Convolutions Reduce Parameters in Convolutional Neural Networks." MENDEL. Vol.
# 28. No. 1. 2022.
# https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/simple_image_classification_with_any_dataset.ipynb
def get_complex_model():     
    l2_decay = 0.000001 #@param {type:"number"}        
    input_shape = (32, 32, 3)

    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), padding='valid',
                    input_shape=input_shape, kernel_regularizer=regularizers.l2(l2_decay)) )
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_decay)) )
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_decay)) )
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(l2_decay)) )
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(l2_decay)) )
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))

    # display the complete architecture
    model.summary()
    return model