import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from glob import glob
import argparse

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.saved_model import tag_constants
import time
from tqdm import tqdm

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as incept_resnet_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_input
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_input

from tensorflow.keras.applications.mobilenet_v2 import decode_predictions as mobilenet_decode
from tensorflow.keras.applications.resnet50 import decode_predictions as resnet_decode
from tensorflow.keras.applications.inception_resnet_v2 import decode_predictions as incept_resnet_decode
from tensorflow.keras.applications.densenet import decode_predictions as densenet_decode
from tensorflow.keras.applications.nasnet import decode_predictions as nasnet_decode

def decode_type(model_name):
    if model_name == 'DenseNet201': return densenet_decode
    elif model_name == 'InceptionV2': return incept_resnet_decode
    elif model_name == 'MobileNetV2': return mobilenet_decode
    elif model_name == 'NASNetLarge': return nasnet_decode
    elif model_name == 'ResNet50': return resnet_decode

def input_type(model_name):
    if model_name == 'DenseNet201': return densenet_input,(224,224)
    elif model_name == 'InceptionV2': return incept_resnet_input, (299,299)
    elif model_name == 'MobileNetV2': return mobilenet_input, (224,224)
    elif model_name == 'NASNetLarge': return nasnet_input, (331,331)
    elif model_name == 'ResNet50': return resnet_input, (224,224)
    
def validation(model_path, images):
    # create batch
    batch_size = 100
    nImages = len(images)
    steps = nImages//batch_size if nImages%batch_size==0 else steps+1
    nCorrect = 0
    
    model_name = model_path.split('/')[-2]
    base_model = model_name.split('_')[0]

    preprocess_input, input_size = input_type(base_model)
    decode_predictions = decode_type(base_model)
    
    # Model Load
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    for step in tqdm(range(steps)):    
        batched_input = np.zeros((batch_size, input_size[0],input_size[1], 3), dtype=np.float32)
        img_path_list = images[step*batch_size:(step+1)*batch_size]

        for i,img_path in enumerate(img_path_list):
            img = image.load_img(img_path, target_size=(input_size[0],input_size[1]))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)
            batched_input[i, :] = x

        batched_input = tf.constant(batched_input,)
        
        output_node = list(infer.structured_outputs.keys())[0]

        result = infer(batched_input)
        preds= decode_predictions(np.array(result[output_node]),top=1)

        imnet_ids = [ids[0][0] for ids in preds]
        class_idx = np.array([imagenet_ids[imnet_id] for imnet_id in imnet_ids])

        gt_batch = gt[step*batch_size:(step+1)*batch_size]

        pred_result = gt_batch==class_idx
        nCorrect+=len(pred_result[pred_result==True])
    
    print (nCorrect,nImages)
    print ((nCorrect/nImages)*100)
    acc = float("{:4.1f}".format((nCorrect/nImages)*100))
    print('Accuracy : {:4.2f}%'.format(acc))
    
    return acc

if __name__=='__main__':
    
    # Arg Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    
    # GT label load
    gt = pd.read_csv('./data/ILSVRC2012_validation_ground_truth.txt',header=None).to_dict()[0]
    gt = list(gt.values())

    mapping = pd.read_csv('./data/ILSVRC2012_mapping.txt',sep=' ',header=None,index_col=0).to_dict()[1]
    imagenet_ids = {v:k for k,v in mapping.items()}

    # Model Info
    model_path = args.model_path
    model_name = model_path.split('/')[-2]

    base_model = model_name.split('_')[0]
    precision_mode = model_name.split('_')[-1]
    
    # Load validation Images
    images = sorted(glob('./data/validation_images/*.*'))
    print ("\n%s validation with %d images\n"%(model_name,len(images)))
    
    # Valdation 
    acc = validation(model_path, images)
    
    # Write Result
    acc_df = pd.read_csv('./result/acc_result.csv',index_col=0)
    acc_df[precision_mode][base_model] = acc
    acc_df.to_csv('./result/acc_result.csv')