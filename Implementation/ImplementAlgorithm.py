#!/usr/bin/python3

"""
author: Abhilash Nair

Standard Implementaion code to run Python scripts in DOSMON's Algorithm Server

#Info 1: Purpose of the code (Eg. Identifying CNI fault and correcting the value)
#Info 2: Author (Eg. Abhilash Nair on (12.02.2022))
#Info 3: Version (1.0.1)

"""

import numpy as np
import datetime as dt
from datetime import datetime
import time
from tb_rest_client.rest_client_pe import *
from tb_rest_client.rest import ApiException
from tb_device_mqtt import TBDeviceMqttClient, TBPublishInfo
import logging
import configparser
import warnings
import pickle
import os
warnings.filterwarnings("ignore")

#%% Load additional libraries necissary to run the model
import sklearn.neighbors._base


#%% API's to read and write data from DOSMON
def download_latest_timeseries_from_client(server):
    """ Download past data and current measurement from ThingsBoard client here """
            
    with RestClientPE(base_url=config[server]['url']) as rest_client:
        try:
            # Auth with credentials
            rest_client.login(username=config[server]['username'], password=config[server]['password'])

            # Get Device details
            deviceDetails = rest_client.get_device_by_id(device_id=config[server]['device_id'])

            # Get get latest timeseries
            data = rest_client.get_latest_timeseries(entity_id=deviceDetails.id)
        # an error has occurred during connection to ThingsBoard
        except ApiException as e:
            print('Connection of the DOSMON server failed. use dummy variables instead')
            data = {'Time': [{'ts': 1682165837368, 'value': '2023-04-22 14:24:48'}],
             'QIN': [{'ts': 1682165837368, 'value': '386.8542785644531'}],
             'CNI': [{'ts': 1682165837368, 'value': '763.2000122070312'}],
             'PHI': [{'ts': 1682165837368, 'value': '7.559374809265137'}]}
            
            logging.exception(e)

    if data:
        return data
    # no data for given time period accessible on ThingsBoard
    else:
        # TODO use suitable logger
        print("No data for last steps on Thingsboard")
        return None


def write_values_to_server(payload, server):
    """ Send data to test ThingsBoard client """
    with RestClientPE(base_url=config[server]['url']) as rest_client:
        try:
            rest_client.login(username=config[server]['username'], password=config[server]['password'])
            # Get Device details
            device_token = rest_client.get_device_credentials_by_device_id(config[server]['device_id'])
            device_token = device_token.credentials_id

            timestamp = int(datetime.timestamp(dt.datetime.today()) * 1000)
            RowToThingsboard = {
                "ts": timestamp,
                "values": payload}

            client = TBDeviceMqttClient(config[server]['mqtt_ip'], device_token)
            client.connect()

            # Sending telemetry without checking the delivery status
            result = client.send_telemetry(RowToThingsboard)
            if result.get() != TBPublishInfo.TB_ERR_SUCCESS:
                # TODO implement suitable logger
                print("Message not sent")

            client.disconnect()

        except ApiException as e:
            logging.exception(e)
    

# %% READ VALUES FROM DOSMON AND THE CONFIG FILES
base_path   = os.path.dirname(os.path.abspath(__file__))
configFile  = os.path.join(base_path,'config.ini') # csv file with plc variable names and adresses
config = configparser.ConfigParser()
config.read(configFile)
data_input_all = download_latest_timeseries_from_client(server='DOSMONRead')

                              
# %% WRITE YOUR ALGORITHM HERE

# Step 1: Select the predictors for the model
data_input_subset = {}
for i in config['Plant']['input_parameters'].split(','):
    data_input_subset[i] = float(data_input_all[i][0]['value'])


CNI_current    = data_input_subset['CNI']
QIN_current    = data_input_subset['QIN']
alarm_array    = list(np.float_(config['Buffer']['previous_alarm'].split(',')))

# Step 2: Prepare the predictor for the Mathematical Model
CNI_difference      = CNI_current - float(config['Buffer']['previous_cni'])
QIN_difference      = QIN_current - float(config['Buffer']['previous_qin'])
data_input_selected = np.array([[CNI_current,
                        QIN_current, 
                        CNI_difference,
                        QIN_difference]])


# Step3: Load model file and generate model predictions
# predict fault
parent_dir            = os.path.abspath(os.path.join(base_path, os.pardir))
modelFile             = os.path.join(parent_dir,'Models',config['Plant']['model_name']) 
fault_detection_model = pickle.load(open(modelFile, 'rb'))
CNI_status            = fault_detection_model.predict(data_input_selected)

# update alarm array
alarm_array.append(float(CNI_status[0]))
alarm_array.pop(0)
    

# correcting CNI values
alpha                = 1.0 #default value of low-passfilter
CNI_difference_limit = 30 #maximum tolerence level for drop in CNI
CNI_corrected        = float(config['Buffer']['previous_cni_corrected']) #previous corrected values

if alarm_array == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]:
    alpha = 0.9999
else:
    if (CNI_current-CNI_corrected)>0:
        alpha = 1
    else:
        if abs(CNI_current-CNI_corrected)<CNI_difference_limit:
            alpha = 0.68
        else:
            alpha = 0.0001
    
CNI_corrected = (1-alpha) * CNI_corrected + alpha * float(data_input_all['CNI'][0]['value'])


# %% SEND DATA TO DOSMON
# Create the payload (in json format) to be sent 
payload = {'CNIRaw':CNI_current,
           'CNICorrect':CNI_corrected,
           'AlarmStatus': float(CNI_status[0])
           }

print(payload)

# payload send
#write_values_to_server(payload, server='DOSMONWrite')
  
# %% SAVE UPDATED VALUES TO CONFIG FILE

# prepare the parameters to be updated in the config files
updated_config_file_values = {'previous_cni':str(CNI_current),
                              'previous_qin':str(QIN_current),
                              'previous_alarm':str(str(alarm_array).replace('[', '').replace(']','')),
                              'previous_cni_corrected':str(CNI_corrected)}

# write the updated values to config files
with open(configFile, 'w') as configfile:
    for i in updated_config_file_values:
        config.set('Buffer', i,updated_config_file_values[i])
    config.write(configfile)
    
