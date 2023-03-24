from keras.layers import Lambda, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input
from keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input
import ray
#from multiprocessing import Pool
#pool = Pool()

img_size = (331,331,3)

def get_features(model_name, model_preprocessor, input_size, data):

    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    feature_maps = feature_extractor.predict(data, verbose=1)
    return feature_maps
# @ray.remote
def a(InceptionV3, inception_preprocessor, img_size, data):
    return get_features(InceptionV3, inception_preprocessor, img_size, data)
# @ray.remote
def b(Xception, xception_preprocessor, img_size, data):
    return get_features(Xception, xception_preprocessor, img_size, data)
# @ray.remote
def c(NASNetLarge, nasnet_preprocessor, img_size, data):
    return get_features(NASNetLarge, nasnet_preprocessor, img_size,data)
# @ray.remote
def d(InceptionResNetV2, inc_resnet_preprocessor, img_size, data):
    return get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, data)

def extact_features(data):
    inception_features = get_features(InceptionV3, inception_preprocessor, img_size, data)
    xception_features = get_features(Xception, xception_preprocessor, img_size, data)
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, img_size, data)
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, data)
    # inception_features = a.remote(InceptionV3, inception_preprocessor, img_size, data)
    # xception_features = a.remote(Xception, xception_preprocessor, img_size, data)
    # nasnet_features = a.remote(NASNetLarge, nasnet_preprocessor, img_size, data)
    # inc_resnet_features = a.remote(InceptionResNetV2,inc_resnet_preprocessor, img_size,data)
    #result1 = pool.apply_async(a, [InceptionV3, inception_preprocessor, img_size, data])    # evaluate "solve1(A)" asynchronously
    #result2 = pool.apply_async(b, [Xception, xception_preprocessor, img_size, data])    # evaluate "solve2(B)" asynchronously
    #result3 = pool.apply_async(c, [NASNetLarge, nasnet_preprocessor, img_size, data])    # evaluate "solve1(A)" asynchronously
    #result4 = pool.apply_async(d, [InceptionResNetV2, inc_resnet_preprocessor, img_size, data])
    #inception_features = result1.get(timeout=10)
    #xception_features = result2.get(timeout=10)
    #nasnet_features = result3.get(timeout=10)
    #inc_resnet_features = result4.get(timeout=10)
    #inception_features, xception_features, nasnet_features, inc_resnet_features = ray.get([inception_features, xception_features, nasnet_features, inc_resnet_features])
    final_features = np.concatenate([inception_features,
                                     xception_features,
                                     nasnet_features,
                                     inc_resnet_features],axis=-1)
    return final_features