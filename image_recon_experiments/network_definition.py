"""
Given the L channel of an Lab image (range [-1, +1]), output a prediction over
the a and b channels in the range [-1, 1].
In the neck of the conv-deconv network use the features from a feature extractor
(e.g. Inception) and fuse them with the conv output.
"""

import sys
sys.dont_write_bytecode = True

#from keras.engine import InputLayer
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, Input
from tensorflow.keras.models import Sequential

from fusion_layer import FusionLayer

def network(input_shape):
    model = Sequential(name="encoder")
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

    model.add(Conv2D(256, (1, 1), activation="relu", padding="same"))
    
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(3, (3, 3), activation="tanh", padding="same"))
    model.add(UpSampling2D((2, 2)))

    return model


def encoder(input_shape):
    # model = Sequential(name="encoder")
    # model.add(Input(shape=input_shape))
    # model.add(Conv2D(32, (5, 5), activation="relu", padding="same", strides=2))
    # model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    # model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=2))
    # model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    # model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=2))
    # model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    # model.add(Conv2D(256, (3, 3), activation="relu", padding="same",strides=3))
    # model.add(Conv2D(15, (5, 5), activation="relu", padding="same"))
    model = Sequential(name="encoder")
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    return model




def decoder(input_shape):
    output_channels = 3
    
    model = Sequential(name="decoder")
    # model.add(Input(shape=input_shape))
    model.add(InputLayer(input_shape=input_shape))
    # model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    # model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    # model.add(Conv2D(3, (3, 3), activation="tanh", padding="same"))
    # model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(256, (1, 1), activation="relu", padding="same"))
    
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2DTranspose(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2DTranspose(64, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(output_channels, (3, 3), activation="tanh", padding="same"))
    model.add(UpSampling2D((2, 2)))

    return model

class Autoencoder:
    # def __init__(self, is_fusion, depth_after_fusion):
    def __init__(self):
        self.network = _build_network()
        # self.encoder = _build_encoder()

        # self.boolfusion=is_fusion
        # if self.boolfusion:
        #     self.fusion = FusionLayer()
        # self.after_fusion = Conv2D(depth_after_fusion, (1, 1), activation="relu")
        # self.decoder = _build_decoder(depth_after_fusion)


    def build_fusion(self, img_l, img_emb):
        image_enc=self.encoder(image_l)
        fusion=self.fusion([image_enc,image_emb])
        fusion=self.after_fusion(fusion)
        return self.decoder(fusion)


    def build(self):
        img_enc = self.encoder()

        # fusion = self.fusion([img_enc, img_emb])
        # fusion = self.after_fusion(fusion)
        fusion = self.after_fusion(img_enc)
        return self.decoder(fusion)


def _build_network():
    input_channels = 3
    output_channels = 3

    # Encoder
    model = Sequential(name="encoder")
    model.add(InputLayer(input_shape=(None, None, input_channels)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    
    model.add(Conv2D(256, (1, 1), activation="relu", padding="same"))
    
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(output_channels, (3, 3), activation="tanh", padding="same"))
    model.add(UpSampling2D((2, 2)))

    return model


def _build_decoder(encoding_depth):
    model = Sequential(name="decoder")
    model.add(InputLayer(input_shape=(None, None, encoding_depth)))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(2, (3, 3), activation="tanh", padding="same"))
    model.add(UpSampling2D((2, 2)))
    return model
