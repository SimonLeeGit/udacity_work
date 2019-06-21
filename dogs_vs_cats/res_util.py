
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.merge import add
from keras.utils import plot_model
import keras.backend as K

class ResNet:
    """Deep Residual Network"""

    def __init__(self):
        pass

    def ResidualBlock(self, input_block, kernels, filters, first_res):
        kernal1, kernal2, kernal3 = kernels
        filter1, filter2, filter3 = filters
        strides = (2,2) if not first_res else (1,1)

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        block1 = Conv2D(filter1, kernal1, strides=strides)(input_block)
        block1 = BatchNormalization(axis=bn_axis)(block1)
        block1 = Activation('relu')(block1)

        block1 = Conv2D(filter2, kernal2, strides=(1,1), padding='same')(block1)
        block1 = BatchNormalization(axis=bn_axis)(block1)
        block1 = Activation('relu')(block1)

        block1 = Conv2D(filter3, kernal3, strides=(1,1))(block1)
        block1 = BatchNormalization(axis=bn_axis)(block1)

        block2 = Conv2D(filter3, kernal3, strides=strides)(input_block)
        block2 = BatchNormalization(axis=bn_axis)(block2)

        block = add([block1, block2])
        block = Activation('relu')(block)

        return block

    def IdentityBlock(self, input_block, kernels, filters):
        kernal1, kernal2, kernal3 = kernels
        filter1, filter2, filter3 = filters

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        block1 = Conv2D(filter1, kernal1, strides=(1,1))(input_block)
        block1 = BatchNormalization(axis=bn_axis)(block1)
        block1 = Activation('relu')(block1)

        block1 = Conv2D(filter2, kernal2, strides=(1,1), padding='same')(block1)
        block1 = BatchNormalization(axis=bn_axis)(block1)
        block1 = Activation('relu')(block1)

        block1 = Conv2D(filter3, kernal3, strides=(1,1))(block1)
        block1 = BatchNormalization(axis=bn_axis)(block1)

        block = add([block1, input_block])
        block = Activation('relu')(block)

        return block

    def BuildModel(self, input_shape, num_outputs, block_shape):
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Check whether channels_last for image data format
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        # Start Block
        input = Input(shape=input_shape)
        block = Conv2D(64, (7,7), strides=(2,2))(input)
        block = BatchNormalization(axis=bn_axis)(block)
        block = Activation('relu')(block)
        block = MaxPool2D((3,3), strides=(2,2))(block)

        # Residual Blocks & Identity Blocks
        for i, num in enumerate(block_shape):
            kernal_size = [1,3,1]
            filters = [64*(i+1), 64*(i+1), 256*(i+1)]

            block = self.ResidualBlock(block, kernal_size, filters, i==0)
            for _ in range(num):
                block = self.IdentityBlock(block, kernal_size, filters)
        
        # End Block
        block = AveragePooling2D((7,1))(block)
        block = Flatten()(block)
        block = Dense(num_outputs)(block)
        output = Activation('softmax')(block)

        # Create a keras model
        model = Model(inputs=input, outputs=output)
        model.summary()

        # Plot model to PNG
        # plot_model(model,to_file='ResNet.png')

        # Compile the model
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def Build_ResNet50(self, input_shape, num_outputs):
        return self.BuildModel(input_shape, num_outputs, (2,3,5,2))

    def Build_ResNet101(self, input_shape, num_outputs):
        return self.BuildModel(input_shape, num_outputs, (2,3,22,2))

    def Build_ResNet152(self, input_shape, num_outputs):
        return self.BuildModel(input_shape, num_outputs, (2,7,35,2))
