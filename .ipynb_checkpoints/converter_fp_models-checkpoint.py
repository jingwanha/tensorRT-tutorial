import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from preprocessing import vgg_preprocess as vgg_preprocessing
from preprocessing import inception_preprocess as inception_preprocessing

import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt

input_size_dict = {
    "MobileNetV2" : (244,244),
    "ResNet50" : (244,244),
    "InceptionV2" : (299,299),
    "DenseNet201" : (244,244),
    "NASNetLarge" : (331,331)
}

def get_basemodel(model_name):
    if model_name == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        return MobileNetV2(include_top=True,weights='imagenet',classes=1000)

    elif model_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import ResNet50
        return ResNet50(include_top=True,weights='imagenet',classes=1000)

    elif model_name=="InceptionV2":
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        return InceptionResNetV2(include_top=True,weights='imagenet',classes=1000)

    elif model_name=="DenseNet201":
        from tensorflow.keras.applications.densenet import DenseNet201
        return DenseNet201(include_top=True,weights='imagenet',classes=1000)

    elif model_name=="NASNetLarge":
        from tensorflow.keras.applications.nasnet import NASNetLarge
        return NASNetLarge(include_top=True,weights='imagenet',classes=1000)
    
if __name__=='__main__':
    
    # BASE MODEL EXPORT
    MODEL_LIST = ["MobileNetV2","ResNet50","InceptionV2","DenseNet201","NASNetLarge"]
    BASE_PATH = "./exported_models/"
    
    # EXPORT BASE MODEL
    for name in MODEL_LIST:
        base_model = get_basemodel(name)
        base_model.save(BASE_PATH+name+"_BASE/1")
        
        
    # EXPORT TRT MODEL(FP32, FP16)
    for name in MODEL_LIST:
        SAVED_MODEL_DIR = BASE_PATH+name+'_BASE/1/'
    
        for mode in ["FP32","FP16"]:
            # print ("CONVERTING %s_%s"%(name,mode))

            if mode == "FP32":
                conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
                                                                               max_workspace_size_bytes=8000000000)
                
                converter = trt.TrtGraphConverterV2(input_saved_model_dir=SAVED_MODEL_DIR,conversion_params=conversion_params)
                converter.convert()
                converter.save(BASE_PATH+name+"_"+mode+'/1')

            elif mode == "FP16":
                conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16,
                                                                               max_workspace_size_bytes=8000000000)
                
                converter = trt.TrtGraphConverterV2(input_saved_model_dir=SAVED_MODEL_DIR,conversion_params=conversion_params)
                converter.convert()
                converter.save(BASE_PATH+name+"_"+mode+'/1')