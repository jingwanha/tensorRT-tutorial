import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from glob import glob
import time
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.saved_model import tag_constants

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as incept_resnet_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_input
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_input

def input_type(model_name):
    if model_name == 'DenseNet201': return densenet_input,(224,224)
    elif model_name == 'InceptionV2': return incept_resnet_input, (299,299)
    elif model_name == 'MobileNetV2': return mobilenet_input, (224,224)
    elif model_name == 'NASNetLarge': return nasnet_input, (331,331)
    elif model_name == 'ResNet50': return resnet_input, (224,224)

def benchmark_tftrt(input_saved_model, batched_input):
    print ("model is loading...")
    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    N_warmup_run = 50
    N_run = 1000
    elapsed_time = []
    
    print ("\nwarm-up iterations ...\n")
    for i in range(N_warmup_run):
        labeling = infer(batched_input)

    print ("\nmain iterations ...\n")
    for i in range(N_run):
        start_time = time.time()
        labeling = infer(batched_input)        
        end_time = time.time()
        
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        
        if i % 50 == 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))
    
    print('Avg elapsed_time: {:4.1f} ms'.format((elapsed_time.mean())*1000))
    print('Throughput: {:.0f} images/s\n\n'.format(N_run * batch_size / elapsed_time.sum()))
    
    return (elapsed_time.mean())*1000, (N_run * batch_size) / elapsed_time.sum()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--batch_size', type=int, default = 1)
    
    args = parser.parse_args()    
    
    model_path = args.model_path
    batch_size = args.batch_size
    
    model_name = model_path.split('/')[-2]
    base_model = model_name.split('_')[0]
    precision_mode = model_name.split('_')[-1]
    
    print ("\nBenchmark Test\n%s with %d batches\n"%(model_name,batch_size))
    
    # Create Batch input
    preprocess_input, input_size = input_type(base_model)
    batched_input = np.zeros((batch_size, input_size[0],input_size[1], 3), dtype=np.float32)
    
    images = glob('./data/validation_images/*.*')
    images = images[0:batch_size]
    
    for i, img_path in enumerate(images):
        img = image.load_img(img_path, target_size=(input_size[0],input_size[1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        x = preprocess_input(x)
        batched_input[i, :] = x

    batched_input = tf.constant(batched_input,)
    print('batched_input shape: ', batched_input.shape)
    
    # Benchmark
    avg_elapsed_time, throughput = benchmark_tftrt(model_path,batched_input)
    
    # Result Save
    elasped_time_df = pd.read_csv('./result/elapsed_time.csv',index_col=0)
    throughput_df = pd.read_csv('./result/throughput.csv',index_col=0)
        
    elasped_time_df[precision_mode][base_model] = float("{:.1f}".format(avg_elapsed_time))
    throughput_df[precision_mode][base_model] = int("{:.0f}".format(throughput))    
    
    elasped_time_df.to_csv('./result/elapsed_time.csv')
    throughput_df.to_csv('./result/throughput.csv')