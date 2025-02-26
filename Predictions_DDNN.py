
'''import numpy as np
import tensorflow as tf
import pathlib
from pickle import load
import time

# This function is for the DDNN model predictions
def pred_func(betas_DNN, Pmax, NoOfSetups, modelname):
    base_path = pathlib.Path().absolute()
    foldername = str(base_path) + '/pAssignModels/DNNmodels/'
    scaler = load(open(foldername + 'scaler.pkl', 'rb'))
    K = betas_DNN.shape[0]
    L = betas_DNN.shape[1]
    nbrOfSetups = betas_DNN.shape[2]
    mu = np.zeros((K+1,L,nbrOfSetups))
    
    for l in range(0, L):
        filename = modelname + str(l+1)
        #DDNN_model = tf.keras.models.load_model(foldername + filename)
        DDNN_model=tf.keras.layers.TFSMLayer(foldername+filename,call_endpoint="serving_default")
#        print('load model %s' % filename)
        start_time = time.perf_counter()
        
        betas =  betas_DNN[:,l,:].T *1000   # linear scale (without kilo)
        
        v = 0.6 
        betas = np.sqrt(Pmax) * ((betas ** v) / np.sum( (betas ** v), axis=1 ).reshape(NoOfSetups,1))
        betas = 10*np.log10(betas)  # dB scale
    
        betas = scaler.transform(betas)
        
        mu[:,l,:]  = DDNN_model.predict(betas).T
        
        stop_time = time.perf_counter() - start_time
        
    return mu , stop_time


# This function is for the DDNN-SI model predictions
def pred_func_extrainput(betas_DNN, Pmax, NoOfSetups, modelname):
    base_path = pathlib.Path().absolute()
    foldername = str(base_path) + '/pAssignModels/DNNmodels-extrainput/'
    scaler = load(open(foldername + 'scaler.pkl', 'rb'))
    K = betas_DNN.shape[0]
    L = betas_DNN.shape[1]
    nbrOfSetups = betas_DNN.shape[2]
    mu = np.zeros((K+1,L,nbrOfSetups))
    
    for l in range(0, L):
        filename = modelname + str(l+1)
        #DDNN_model = tf.keras.models.load_model(foldername + filename)
        DDNN_model=tf.keras.layers.TFSMLayer(foldername+filename,call_endpoint="serving_default")
#        print('load model %s' % filename)
        start_time = time.perf_counter()
        
        betas =  betas_DNN[:,l,:].T *1000   # linear scale (no kilo)
        
        betas_to_all = betas_DNN * 1000    # linear scale (no kilo)
        
        v = 0.6                     # Fractional power allocation factor
        extraInput = np.sqrt(Pmax) * ((betas ** v) / np.sum(betas_to_all ** v, axis=1 ).T)
        extraInput = 10*np.log10(extraInput)    # dB scale
        
        betas = np.sqrt(Pmax) * ((betas ** v) / np.sum( (betas ** v), axis=1 ).reshape(NoOfSetups,1))
        betas = 10*np.log10(betas)  # dB scale
    
        betas = scaler.transform(betas)
        extraInput = scaler.transform(extraInput)
        DNNinput = np.concatenate((betas, extraInput), axis=1)
        
        mu[:,l,:]  = DDNN_model.predict(DNNinput).T
        
        stop_time = time.perf_counter() - start_time
        
    return mu , stop_time


# This function is for the DDNN model predictions without heuristic and total power adjustment
def pred_func_without(betas_DNN, Pmax, NoOfSetups, modelname):
    base_path = pathlib.Path().absolute()
    foldername = str(base_path) + '/pAssignModels-without/DNNmodels/'
    scaler = load(open(foldername + 'scaler.pkl', 'rb'))
    K = betas_DNN.shape[0]
    L = betas_DNN.shape[1]
    nbrOfSetups = betas_DNN.shape[2]
    mu = np.zeros((K,L,nbrOfSetups))
    
    for l in range(0, L):
        filename = modelname + str(l+1)
        DDNN_model = tf.keras.models.load_model(foldername + filename)
#        print('load model %s' % filename)
        start_time = time.perf_counter()
        
        betas =  betas_DNN[:,l,:].T *1000   # linear scale (without kilo)
        
        betas = 10*np.log10(betas)  # dB scale
    
        betas = scaler.transform(betas)
        
        mu[:,l,:]  = DDNN_model.predict(betas).T
        
        stop_time = time.perf_counter() - start_time
        
    return mu , stop_time'''


import numpy as np
import tensorflow as tf
import pathlib
from pickle import load
import time
import os

# This function is for the DDNN model predictions
def pred_func(betas_DNN, Pmax, NoOfSetups, modelname):
    base_path = pathlib.Path().absolute()
    foldername = os.path.join(base_path, 'pAssignModels', 'DNNmodels')
    scaler_path = os.path.join(foldername, 'scaler.pkl')
    
    # Load scaler if it exists
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler = load(open(scaler_path, 'rb'))
    
    K = betas_DNN.shape[0]
    L = betas_DNN.shape[1]
    nbrOfSetups = betas_DNN.shape[2]
    mu = np.zeros((K+1, L, nbrOfSetups))
    
    for l in range(L):
        filename = f"{modelname}{l+1}.keras"
        model_path = os.path.join(foldername, filename)
        
        # Check if model file exists
        if not os.path.isfile(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping this setup.")
            continue
        
        DDNN_model = tf.keras.models.load_model(model_path)
        start_time = time.perf_counter()
        
        betas = betas_DNN[:, l, :].T * 1000  # Linear scale (without kilo)
        
        v = 0.6 
        betas = np.sqrt(Pmax) * ((betas ** v) / np.sum((betas ** v), axis=1).reshape(NoOfSetups, 1))
        betas = 10 * np.log10(betas)  # dB scale
        
        betas = scaler.transform(betas)
        
        mu[:, l, :] = DDNN_model.predict(betas).T
        
        stop_time = time.perf_counter() - start_time
        
    return mu, stop_time


# This function is for the DDNN-SI model predictions
def pred_func_extrainput(betas_DNN, Pmax, NoOfSetups, modelname):
    base_path = pathlib.Path().absolute()
    foldername = os.path.join(base_path, 'pAssignModels', 'DNNmodels')
    scaler_path = os.path.join(foldername, 'scaler.pkl')
    
    # Load scaler if it exists
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler = load(open(scaler_path, 'rb'))
    
    K = betas_DNN.shape[0]
    L = betas_DNN.shape[1]
    nbrOfSetups = betas_DNN.shape[2]
    mu = np.zeros((K+1, L, nbrOfSetups))
    
    for l in range(L):
        filename = f"{modelname}{l+1}"
        model_path = os.path.join(foldername, filename)
        
        # Check if model file exists
        if not os.path.isfile(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping this setup.")
            continue
        
        DDNN_model = tf.keras.models.load_model(model_path)
        start_time = time.perf_counter()
        
        betas = betas_DNN[:, l, :].T * 1000
        betas_to_all = betas_DNN * 1000
        
        v = 0.6
        extraInput = np.sqrt(Pmax) * ((betas ** v) / np.sum(betas_to_all ** v, axis=1).T)
        extraInput = 10 * np.log10(extraInput)  # dB scale
        
        betas = np.sqrt(Pmax) * ((betas ** v) / np.sum((betas ** v), axis=1).reshape(NoOfSetups, 1))
        betas = 10 * np.log10(betas)  # dB scale
        
        betas = scaler.transform(betas)
        extraInput = scaler.transform(extraInput)
        DNNinput = np.concatenate((betas, extraInput), axis=1)
        
        mu[:, l, :] = DDNN_model.predict(DNNinput).T
        
        stop_time = time.perf_counter() - start_time
        
    return mu, stop_time


# This function is for the DDNN model predictions without heuristic and total power adjustment
def pred_func_without(betas_DNN, Pmax, NoOfSetups, modelname):
    base_path = pathlib.Path().absolute()
    foldername = os.path.join(base_path, 'pAssignModels', 'DNNmodels')
    scaler_path = os.path.join(foldername, 'scaler.pkl')
    
    # Load scaler if it exists
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler = load(open(scaler_path, 'rb'))
    
    K = betas_DNN.shape[0]
    L = betas_DNN.shape[1]
    nbrOfSetups = betas_DNN.shape[2]
    mu = np.zeros((K, L, nbrOfSetups))
    
    for l in range(L):
        filename = f"{modelname}{l+1}"
        model_path = os.path.join(foldername, filename)
        
        # Check if model file exists
        if not os.path.isfile(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping this setup.")
            continue
        
        DDNN_model = tf.keras.models.load_model(model_path)
        start_time = time.perf_counter()
        
        betas = betas_DNN[:, l, :].T * 1000  # Linear scale (without kilo)
        betas = 10 * np.log10(betas)  # dB scale
        
        betas = scaler.transform(betas)
        
        mu[:, l, :] = DDNN_model.predict(betas).T
        
        stop_time = time.perf_counter() - start_time
        
    return mu, stop_time
