
import tensorflow as tf
import numpy as np
import os


from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model 



def Model(input_shape = (128, 128, 3), n_filters = 32, n_classes = 256, dropout_prob = 0.3):

     
    
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """

    inputs = Input(input_shape)
    
    cblock1 = Encoder(inputs, n_filters)
    
    cblock2 = Encoder(cblock1[0], n_filters*2)
    cblock3 = Encoder(cblock2[0], n_filters *4)
    cblock4 = Encoder(cblock3[0], n_filters * 8, dropout_prob) # Include a dropout of 0.3 for this layer
    
    cblock5 = Encoder(cblock4[0], n_filters*16, dropout_prob, max_pooling=None) 
    

    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    ublock6 = Decoder(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = Decoder(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = Decoder(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = Decoder(ublock8, cblock1[1],  n_filters)
    
    conv9 = Conv2D(n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='he_normal')(ublock9)

    
    outputs = Conv2D(n_classes, 1, padding='same')(conv9)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def Encoder(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """


    convolve = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    convolve  = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(convolve)




    if dropout_prob > 0:
        
        convolve = Dropout(dropout_prob)(convolve)
        
         
        
    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2, 2))(convolve)
        
    else:
        next_layer = convolve
        
    skip_connection = convolve
    
    return next_layer, skip_connection



def Decoder(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """
    upsample = Conv2DTranspose(
                 n_filters,    # number of filters
                 (3, 3),    # Kernel size
                 strides=(2, 2),
                 padding='same')(expansive_input)
    
    # Merge the previous output and the contractive_input
    merge = concatenate([upsample, contractive_input], axis=3)

    convolve = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(merge)


    convolve = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(convolve)
  
    
    return convolve




model = Model()
print(model.layers)
print(model.summary())

plot_model(model, to_file='model.png') 