#@title Licensed under the Apache License, Version 2.0 (the "License");
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

# Run prelininary imports
import sys
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from codecarbon import EmissionsTracker
from codecarbon import OfflineEmissionsTracker
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import Callback
from datetime import datetime

import argparse
import cct
import models
import subprocess
import os

parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('--sect', type=str, default='init', required=False, help='Selects the section to run')
args = parser.parse_args()

SLEEP_TIME_SECONDS = 5

ALUMET_RESULT_FILENAME = "./Results-CSV/"+datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+"_alumet-output.csv"

ONE_JOULE_TO_KWH = 2.77778e-7

alumet_command = ["./bin/alumet-local-agent"]

##############################################
############# Sections Outline ###############
##############################################

# 1a: Standard Run
# 1b: Basic use of CodeCarbon
# 1c: Energy is all we need: tracking energy
# 1d: Using Alumet with perf-events
# 2a: Getting energy information when training

sections = ['1a', '1b', '1c', '1d',
            '2a']

if args.sect not in sections:
    print ('Incorrect section name. Please choose a section between 1[a-d] or 2[a]. Example: python session.py --sect 1c')
    sys.exit()


# Read CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

num_classes = len(class_names)

##############################################
######### Section 1a: Standard Run ###########
##############################################
if args.sect == '1a':
    start_1a = time.time()
    # Getting our simple convolutional model
    model = models.get_simple_model()

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Running the training loop
    history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
                
    # Evaluating the trained model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    end_1a = time.time()
    time_1a = end_1a - start_1a

    # Printing some results
    print('Session 1a completed')
    print('Test accuracy: '+ str(test_acc))
    print('Processing time: '+ str(time_1a) + ' seconds')    
    sys.exit()

##############################################
### Section 1b: Basic use of CodeCarbon    ###
##############################################
elif args.sect == '1b':
    start_1b = time.time()
    # Getting our simple convolutional model
    model = models.get_simple_model()

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Creating the tracker object
    # Country ISO codes can be found on Wikipedia
    # https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
    # Tracker initialization parameters:
    # https://github.com/mlco2/codecarbon/blob/96c1ce15dbf33eaaaa378d3104bde64bfc9f1416/codecarbon/emissions_tracker.py#L157
    tracker = OfflineEmissionsTracker(country_iso_code='BRA', log_level='debug')   

    # Start Tracking 
    tracker.start()

    # Running the training loop
    history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

    # Stop the tracking
    # tracker.stop() returns CO2 emissions in kilograms
    # Source: https://github.com/mlco2/codecarbon/blob/96c1ce15dbf33eaaaa378d3104bde64bfc9f1416/codecarbon/emissions_tracker.py#L408
    emissions = tracker.stop()
                
    # Evaluating the trained model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    end_1b = time.time()
    time_1b = end_1b - start_1b

    # Printing some results
    print('Session 1b completed')
    print('Test accuracy: '+ str(test_acc))
    print('Processing time: '+ str(time_1b) + ' seconds')
    print('Emissions: '+ str(emissions) + ' KgCO2e') 

    ## Q1: How much is the instrumentation (tracking) overhead?
    # The overhead is how much additional time it takes to process the
    # instrumented training, when compared to the same training without tracking
    # this can be calculated by overhead = time_tracked / time_untracked    
    sys.exit()

##############################################
################ Section 1c: #################
### Energy is all we need: tracking energy ###
##############################################
elif args.sect == '1c':
    start_1c = time.time()
    # Getting our simple convolutional model
    model = models.get_simple_model()    

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Creating the tracker object
    # Country ISO codes can be found on Wikipedia
    # https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
    tracker = OfflineEmissionsTracker(country_iso_code='BRA', log_level='debug')

    # Start Tracking 
    tracker.start()

    # Running the training loop
    history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

    # Stop the tracking
    # tracker.stop() returns CO2 emissions in kilograms
    # Source: https://github.com/mlco2/codecarbon/blob/96c1ce15dbf33eaaaa378d3104bde64bfc9f1416/codecarbon/emissions_tracker.py#L408
    emissions = tracker.stop()
                
    # Evaluating the trained model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    end_1c = time.time()
    time_1c = end_1c - start_1c    

    # Printing some results
    print('Session 1c completed')
    print('Test accuracy: '+ str(test_acc))
    print('Processing time: '+ str(time_1c) + ' seconds')
    print('Total CPU energy consumption: ' + str(tracker._total_cpu_energy.kWh) + ' kWh')
    print('Total RAM energy consumption: ' + str(tracker._total_ram_energy.kWh) + ' kWh')
    print('Total Energy consumption: ' + str(tracker._total_energy.kWh) + ' kWh')
    print('Emissions by CodeCarbon: '+ str(emissions) + ' kgCO2eq')

    print('\nAnswer of questions Q2 and Q3')
    ## Q2: How to take into account the power usage efficiency (PUE) of a datacenter?
    # Calculate a new total energy consumption taking the tracker's output as a base
    # Use a PUE value of 1.1
    power_usage_efficiency = 1.1

    #### Q2 answer code goes here ####
    q2_answer = tracker._total_energy.kWh * power_usage_efficiency
    ##################################    
    print('My calculated total energy consumption (with PUE): ' + str(q2_answer) + ' kWh')

    ## Q3: How to properly calculate CO2 emissions using datacenter data?
    # Calculate a new CO2 emissions taking the Q2 output as a base
    # Use the below two use cases: dc1 and dc2 datacenters
    # The carbon intensity (CI) is the grams (i.e., g, not Kg) of CO2 emitted by kilowatt hour
    # CFE% stands for the percentage of carbon free energy
    # a CFE of 10% means that 10% of the energy has a CI of 0
    # the remaining 90% emits CO2 according to the corresponding CI

    #Data center data
    #Name    CFE%    CI (gCO2e/kWh)
    #dc1     11%	  746
    #dc2     91%	  127
    
    #### Q3 answer code goes here ####
    dc_data: dict = {
                    'dc1':{'CFE':11, 'CI':746},
                    'dc2':{'CFE':91, 'CI':127}
                    }
    q3_answer_dc1 = q2_answer * (1 - (dc_data['dc1']['CFE'] / 100)) * (dc_data['dc1']['CI'] / 100)
    q3_answer_dc2 = q2_answer * (1 - (dc_data['dc2']['CFE'] / 100)) * (dc_data['dc2']['CI'] / 100)    
    ##################################   
    print('My calculated Emissions (dc1): '+ str(q3_answer_dc1) + ' kgCO2eq')
    print('My calculated Emissions (dc2): '+ str(q3_answer_dc2) + ' kgCO2eq')
    sys.exit()

##############################################
################ Section 1d: #################
####### Using Alumet with perf-events ########
##############################################
elif args.sect == '1d':
    start_1c = time.time()
    # Getting our simple convolutional model
    model = models.get_simple_model()    

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

   
    # Start tracking by lauching Alumet as a separate process
    alumet = subprocess.Popen(alumet_command)

    # Make a short sleep to let Alumet to start
    time.sleep(SLEEP_TIME_SECONDS)

    # Running the training loop
    history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
    
    # Make a short sleep just to better illustrate the power curve
    time.sleep(SLEEP_TIME_SECONDS)

    # Stop the tracking by terminating the Alumet process  
    alumet.terminate()      

    # Rename Alumet's default csv file for better experiment control    
    os.rename("./alumet-output.csv", ALUMET_RESULT_FILENAME)
                
    # Evaluating the trained model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    end_1c = time.time()
    time_1c = end_1c - start_1c  

    # Printing some results
    print('Session 1c completed')
    print('Test accuracy: '+ str(test_acc))
    print('Processing time: '+ str(time_1c) + ' seconds')   
    #print('Total Energy consumption: ' + str(tracker._total_energy.kWh) + ' kWh')
    #print('Emissions by CarbonTracker: '+ str(emissions) + ' KgCO2e')

    ## Q4: How is the power consumption of the CPU?
    # Use Alumet's output file (whose path is described by the
    # ALUMET_RESULT_FILENAME variable) to get a plot curve of the CPU power
    # consumption. 
    
    # Hint 1: for a time period t, we can calculate the power in watts by
    # dividing the energy consumed (in joules) during the time period t by the
    # length of t in seconds. By default Alumet collects the energy consumption
    # at one second intervals. This may facilidate your task.

    # Hint 2: You can use Pandas to read Alumet's output
    # Hint 3: You can use Seaborn's lineplot function to make the plot

    #### Q4 answer code goes here ####
    # Be mindful of the identation level

    # Get alumet output data
    df_alumet_output = pd.read_csv(ALUMET_RESULT_FILENAME, sep=';')
    df_alumet_output_cpu = df_alumet_output.loc[df_alumet_output['domain'] == 'pp0']
    df_alumet_output_platform = df_alumet_output.loc[df_alumet_output['domain'] == 'platform']
    total_cpu_energy_kWh = df_alumet_output_cpu['value'].sum() * ONE_JOULE_TO_KWH
    total_pletform_energy_kWh = df_alumet_output_platform['value'].sum() * ONE_JOULE_TO_KWH

    # Printing the energy consumption collected by Alumet
    print('Total CPU energy consumption: ' + str(total_cpu_energy_kWh) + ' kWh')
    print('Total platform energy consumption: ' + str(total_pletform_energy_kWh) + ' kWh')


    # Plot some nice power curves The proper way to get the power is to divide
    # the measured energy by measurement the time interval Here i exploit the
    # fact that Alumet takes measurements at one second intervals by default. So
    # the power in Watts is the consumed energy during this one second interval     
    g = sns.lineplot(x='timestamp', y='value', data=df_alumet_output_cpu)
    g.set_ylabel('Power (W)')
    g.set_xlabel('Timestamp')
    plt.xticks(rotation=90)
    plt.show()

    ##################################   

    ## Q5: How can we calculate the emissions with Alumet's data?
    # Hint: use the Electricity Maps website
    # (https://app.electricitymaps.com/map) to know the carbon intensity of  the
    # electricity in your region. Then multiply the consumed energy by the
    # carbon intensity. Pay extra attention to the units to avoid erroneous
    # multiplication

    #### Q5 answer code goes here ####
    # Be mindful of the identation level

    carbon_intensity_brasila_kgco2_per_kwh = 0.112
    total_emissions_platform_kgco2 = total_pletform_energy_kWh * carbon_intensity_brasila_kgco2_per_kwh
    print('Emissions: '+ str(total_emissions_platform_kgco2) + ' kgCO2eq')

    ##################################  
    
    sys.exit()



##############################################
################ Section 2a: #################
## Getting energy information when training ##
##############################################
elif args.sect == '2a':    
    start_2a = time.time()      

    # Creating a callback method to collect data while training
    class MyTrainingCallBack(Callback):
        def __init__(self, codecarbon_tracker):
            self.codecarbon_tracker = codecarbon_tracker
            pass

        ## Q6: How to stop training in an epoch when we pass a energy cap?
        # Use the energy measured at section 1b as an energy cap for the
        # training 
        # 
        # Hint: variable to tell TF to stop training: self.model.stop_training
        # (True or False)       
        def on_epoch_end(self, epoch, logs=None):
            self.codecarbon_tracker.flush()

            # Energy measured in the 1b run on my laptop 
            energy_cap_kwh = 0.001071591612688175
            
            # Getting the total energy consumption from the tracker
            train_total_energy = self.codecarbon_tracker._total_energy.kWh

            # Checking if we pass the energy cap at the end of the epoch
            if train_total_energy >= energy_cap_kwh:
                # command to tell TF to stop training
                self.model.stop_training = True            

        ## Q7: How to stop training in a **batch** when we pass a energy cap?
        # Use the energy measured at section 1b as an energy cap for the
        # training
        #   
        # Useful resources: Custom callbacks:
        # https://www.tensorflow.org/guide/keras/custom_callback 
        # 
        # Hint: use self.codecarbon_tracker._measure_power_and_energy() instead
        # of self.codecarbon_tracker.flush() to avoid IO overhead
        ## Q7: What happens if you don't call _measure_power_and_energy() or flush()?
        def on_batch_end(self, batch, logs=None):
            # Energy measured in the 1b run on my laptop 
            energy_cap_kwh = 0.001071591612688175
            
            # Actively calling the trackers's function to get energy values
            # Otherwise the tracker will get energy values only every 15 seconds
            # (default value). We could use flush but flush performs IO
            # operations and this would result in a larger overhead
            self.codecarbon_tracker._measure_power_and_energy()

            # Getting the total energy consumption from the tracker
            train_total_energy = self.codecarbon_tracker._total_energy.kWh            

            # Checking if we pass the energy cap at the end of the batch
            if train_total_energy >= energy_cap_kwh:
                # command to tell TF to stop training
                self.model.stop_training = True


    # Small label reshape to fit the CCT model
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)    

    # Model obtained from Hassani, Ali, et al. "Escaping the big data paradigm
    # with compact transformers." arXiv preprint arXiv:2104.05704 (2021).
    # https://github.com/keras-team/keras-io/blob/master/examples/vision/cct.py
    # https://keras.io/examples/vision/cct/
    model = cct.create_cct_model()

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True, label_smoothing=0.1
                  ),                  
                  metrics=['accuracy'])

    # Creating the tracker object
    # Country ISO codes can be found on Wikipedia
    # https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
    tracker = OfflineEmissionsTracker(country_iso_code='BRA', log_level='error')

    # Initializing my call back object to be used during training
    my_callback = MyTrainingCallBack(tracker)

    # Start Tracking 
    tracker.start()

    # Running the training loop
    history = model.fit(train_images, train_labels, epochs=30, batch_size=128, 
                    validation_data=(test_images, test_labels), callbacks=[my_callback])

    # Stop the tracking
    # tracker.stop() returns CO2 emissions in kilograms
    # Source: https://github.com/mlco2/codecarbon/blob/96c1ce15dbf33eaaaa378d3104bde64bfc9f1416/codecarbon/emissions_tracker.py#L408
    emissions = tracker.stop()
                
    # Evaluating the trained model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    end_2a = time.time()
    time_2a = end_2a - start_2a    

    # Printing some results
    print('Session 2a completed')
    print('Test accuracy: '+ str(test_acc))
    print('Processing time: '+ str(time_2a) + ' seconds')
    print('Total CPU energy consumption: ' + str(tracker._total_cpu_energy.kWh) + ' kWh')
    print('Total RAM energy consumption: ' + str(tracker._total_ram_energy.kWh) + ' kWh')
    print('Total Energy consumption: ' + str(tracker._total_energy.kWh) + ' kWh')
    print('Emissions by CodeCarbon: '+ str(emissions) + ' kgCO2eq')   
    
    sys.exit()