
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import AveragePooling2D
from keras.layers import merge
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import plot_model
import keras.backend as K

class ResNet:
    """Deep Residual Network"""

    def __init__(self):
        pass

    def ResidualBlock(self, input_block, kernels, filters, strides):
        kernal1, kernal2, kernal3 = kernels
        filter1, filter2, filter3 = filters

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        block1 = Conv2D(filter1, kernal1, strides=strides)(input_block)
        block1 = BatchNormalization(axis=bn_axis)(block1)
        block1 = Activation('relu')(block1)

        block1 = Conv2D(filter2, kernal2, strides=strides, padding='same')(block1)
        block1 = BatchNormalization(axis=bn_axis)(block1)
        block1 = Activation('relu')(block1)

        block1 = Conv2D(filter3, kernal3, strides=strides)(block1)
        block1 = BatchNormalization(axis=bn_axis)(block1)

        block2 = Conv2D(filter3, kernal1, strides=strides)(input_block)
        block2 = BatchNormalization(axis=bn_axis)(block2)

        block = merge([block1, block2], model='sum')
        block = Activation('relu')(block)

        return block

    def IdentityBlock(self, input_block, kernels, filters, strides):
        kernal1, kernal2, kernal3 = kernels
        filter1, filter2, filter3 = filters

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        block1 = Conv2D(filter1, kernal1, strides=strides)(input_block)
        block1 = BatchNormalization(axis=bn_axis)(block1)
        block1 = Activation('relu')(block1)

        block1 = Conv2D(filter2, kernal2, strides=strides, padding='same')(block1)
        block1 = BatchNormalization(axis=bn_axis)(block1)
        block1 = Activation('relu')(block1)

        block1 = Conv2D(filter3, kernal3, strides=strides)(block1)
        block1 = BatchNormalization(axis=bn_axis)(block1)

        block = merge([block1, input_block], model='sum')
        block = Activation('relu')(block)

        return block

    def BuildModel(self, input_shape, num_outputs, block_shape):
        input = Input(shape=input_shape)

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        # Start Block
        block = Conv2D(64, (7,7), strides=(2,2))(input)
        block = BatchNormalization(axis=bn_axis)(block)
        block = Activation('relu')(block)
        block = MaxPool2D((3,3), strides=(2,2))(block)

        # Residual Blocks & Identity Blocks
        for i, num in enumerate(block_shape):
            kernal_size = [1,3,1]
            filters = [64 * (i+1), 64 * (i+1), 256 * (i+1)]
            strides = (2,2) if i > 0 else (1,1)

            block = self.ResidualBlock(block, kernal_size, filters, strides=strides)

            for _ in range(num):
                block = self.IdentityBlock(block, kernal_size, filters, strides=strides)
        
        # End Block
        block = AveragePooling2D((7,1))(block)
        block = Flatten()(block)
        block = Dense(num_outputs)(block)
        output = Activation('softmax')(block)

        # Create a keras model
        model = Model(inputs=input, outputs=output)
        model.summary()

        # Plot model to PNG
        plot_model(model,to_file='ResNet.png')

        # Compile the model
        model.compile(optimizer='sgd',loss='categorical_crossentropy')
        
        return model

    def Build_ResNet50(self, input_shape, num_outputs):
        return self.BuildModel(input_shape, num_outputs, (2,3,5,2))

    def Build_ResNet101(self, input_shape, num_outputs):
        return self.BuildModel(input_shape, num_outputs, (2,3,22,2))

    def Build_ResNet152(self, input_shape, num_outputs):
        return self.BuildModel(input_shape, num_outputs, (2,7,35,2))
