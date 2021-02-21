import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_input
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_input

from glob import glob
        
input_size_dict = {
    "MobileNetV2" : (244,244),
    "ResNet50" : (244,244),
    "InceptionV2" : (299,299),
    "DenseNet201" : (244,244),
    "NASNetLarge" : (331,331)
}
    
    
if __name__=='__main__':
    
    # BASE MODEL EXPORT
    MODEL_LIST = ["MobileNetV2","ResNet50","InceptionV2","DenseNet201","NASNetLarge"]
    BASE_PATH = "./exported_models/"                
            
    # EXPORT INT8 MODEL
    # Data calibration의 경우 imagenet 데이터셋을 이용하여 클래스 당 1장의 이미지 (총 1000장)을 사용하여  calibration 수행
    data_directory = "./data/calibration_images/"
    calibration_files = glob(data_directory+'*.*')
    print('There are %d calibration files. \n%s\n%s\n...'%(len(calibration_files), calibration_files[0], calibration_files[-1]))
        
#    for name in MODEL_LIST:
    for name in ["DenseNet201"]:
        mode = 'INT8'
        SAVED_MODEL_DIR = BASE_PATH+name+'_BASE/1/'
        
        calibration_input = np.zeros((len(calibration_files), input_size_dict[name][0], input_size_dict[name][1], 3), dtype=np.float32)
        
        for i,img_path in enumerate(calibration_files):

            img = image.load_img(img_path, target_size=(input_size_dict[name][0], input_size_dict[name][1]))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            if name == "MobileNetV2" : x = mobilenet_input(x)
            if name == "ResNet50" : x = resnet_input(x)
            if name == "InceptionV2" : x = inception_input(x)
            if name == "DenseNet201" : x = densenet_input(x)
            if name == "NASNetLarge" : x = nasnet_input(x)
            
            calibration_input[i, :] = x
        
        calibration_input = tf.constant(calibration_input)
        print('calibration_input shape: ', calibration_input.shape)
        
        print ("\n\nCONVERTING %s_%s\n\n"%(name,mode))
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.INT8,
                                                                       max_workspace_size_bytes=8000000000,
                                                                       use_calibration=True)
        
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=SAVED_MODEL_DIR,conversion_params=conversion_params)
        
        def calibration_input_fn():
            yield (calibration_input, )
    
        converter.convert(calibration_input_fn=calibration_input_fn)
        converter.save(BASE_PATH+name+"_"+mode+'/1')