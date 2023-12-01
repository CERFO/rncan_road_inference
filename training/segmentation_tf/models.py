import tensorflow as tf

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.layers import Reshape, Permute, add, multiply

import tensorflow.keras.backend as K
import tensorflow_addons as tfa

from tensorflow.keras.losses import binary_crossentropy
from keras_unet_collection.losses import focal_tversky, dice, dice_coef

from keras_unet_collection.transformer_layers import patch_extract, patch_embedding, SwinTransformerBlock, patch_merging, patch_expanding

strategy = tf.distribute.MirroredStrategy(
   devices=["GPU:0", "GPU:1", "GPU:2"],
   cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
   )


###############################################################################################################
# coswin
# https://arxiv.org/ftp/arxiv/papers/2201/2201.03178.pdf
###############################################################################################################
def swin_branch(X, embed_dim, patch_size, num_patch_x, num_patch_y, num_heads, window_size, num_mlp,
                qkv_bias, qk_scale, mlp_drop_rate, attn_drop_rate, proj_drop_rate, drop_path_rate):
    # Patch extraction
    X = patch_extract(patch_size)(X)

    # Embed patches to tokens
    X = patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    shift_size = window_size // 2
    X = SwinTransformerBlock(dim=embed_dim, num_patch=(num_patch_x, num_patch_y), num_heads=num_heads, 
                                 window_size=window_size, shift_size=shift_size, num_mlp=num_mlp, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, drop_path_prob=drop_path_rate)(X)
    
    # Patch merging
    X = patch_expanding(num_patch=(num_patch_x, num_patch_y),
                        #embed_dim=embed_dim, upsample_rate=patch_size[0], return_vector=False)(X)
                        embed_dim=embed_dim, upsample_rate=patch_size[0], return_vector=False)(X)
    
    return X


def context_guided_filter(X_skip, X_deep, kernel_init='he_normal', kernel_reg='l2'):
    X_avg = tf.math.reduce_mean(X_deep, axis=-1)
    X_max = tf.math.reduce_max(X_deep, axis=-1)
    
    X_avg = tf.expand_dims(X_avg, axis=-1)
    X_max = tf.expand_dims(X_max, axis=-1)
    
    X = Concatenate()([X_avg, X_max])
    X = Conv2D(1, (1, 1), padding='same', use_bias=False,
            kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(X)
    X = Activation('sigmoid')(X)
    
    X = multiply([X_skip, X])
    X = add([X, X_deep])
    
    return X


def res_block(x, n_filters, n_strides=1, act='swish',
        kernel_init='he_normal', kernel_reg='l2'):
    # conv
    x1 = Conv2D(n_filters, (1,1), n_strides, padding='same',
            kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(act)(x1)
    x1 = Conv2D(n_filters, (3,3), padding='same',
            kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x1)
    x1 = BatchNormalization()(x1)
    # prepare residuals
    if n_strides != 1:
        x = Conv2D(n_filters, (1,1), n_strides,
                kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x)
        x = BatchNormalization()(x)
    if x.shape[-1] != x1.shape[-1]:
        x = Conv2D(n_filters, (1,1), padding='same',
                kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x)
        x = BatchNormalization()(x)
    # residuals
    x = add([x, x1])
    x = Activation(act)(x)
    x = SpatialDropout2D(0.2)(x)
    return x


def upsampling_block(x, n_filters, upsample_type='conv2dtranspose', bn=True, act_layer=True, act='swish',
        kernel_init='he_normal', kernel_reg='l2'):
    if upsample_type == 'upsampling2d':
        x = UpSampling2D((2,2))(x)
        x = Conv2D(n_filters, (1,1),  padding='same',
                kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x)
    elif upsample_type == 'conv2dtranspose':
        x = Conv2DTranspose(n_filters, (3,3), strides=2, padding='same',
                kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)(x)
    if bn:
        x = BatchNormalization()(x)
    if act_layer:
        x = Activation(act)(x)
    return x


def get_coswin_model(img_shape=(128,128,3), n_labels=1, cnn_act='swish',
        kernel_init='he_normal', kernel_reg=tf.keras.regularizers.l2(1e-4), lr=7e-4, to_compile=True):

    # config
    n_filters_begin = 48
    n_filters = [48, 96, 192, 384]
    #n_filters = [32, 64, 128, 256]
    
    patch_size = (4,4)
    num_heads=[4, 8, 16, 32]
    window_size=[4, 2, 2, 2]
    num_mlp=2
    qkv_bias = True
    qk_scale = None
    mlp_drop_rate = 0.02 # Droupout after each MLP layer
    attn_drop_rate = 0.02 # Dropout after Swin-Attention
    proj_drop_rate = 0.02 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    drop_path_rate = 0.02 # Drop-path within skip-connections
    
    depth = 3
    
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    with strategy.scope():
    
    
        # Input tensor
        inputs = Input(img_shape)
        X = inputs
        
        # Encoder
        skips = []
        for i in range(depth):
            # config
            embed_dim = n_filters[i]*4
            num_patch_x = X.shape[1]//patch_size[0]
            num_patch_y = X.shape[2]//patch_size[1]
            n_heads = num_heads[i]
            w_size = window_size[i]
            
            # Swin branch
            X_swin = swin_branch(X, embed_dim, patch_size, num_patch_x, num_patch_y, n_heads, w_size, num_mlp,
                        qkv_bias, qk_scale, mlp_drop_rate, attn_drop_rate, proj_drop_rate, drop_path_rate)
            # CNN branch
            dropout = True if i > 0 else False
            X_cnn = rec_res_block(X, n_filters[i], act=cnn_act, dropout=dropout, kernel_reg=kernel_reg)
            # Add features
            X = add([X_swin, X_cnn])
            # Append skip connections
            skips.append(X)
            # Dowsample features
            X = MaxPooling2D((2,2))(X)
        
        # bottleneck
        X_deep = res_btlnck(X, n_filters[-1], act=cnn_act)
        
        # Decoder
        X = res_block(X_deep, n_filters[-2], act=cnn_act, kernel_init=kernel_init, kernel_reg=kernel_reg)
        for i in reversed(range(depth)):
            # Get skip connection and deep features
            X_skip = skips.pop()
            # Upsampling
            X = upsampling_block(X, n_filters[i], kernel_init=kernel_init, kernel_reg=kernel_reg)
            X_deep = upsampling_block(X_deep, n_filters[i], kernel_init=kernel_init, kernel_reg=kernel_reg)
            # Context guided filter
            X_cf = context_guided_filter(X_skip, X_deep, kernel_init=kernel_init, kernel_reg=kernel_reg)
            # Decoding stage
            X = add([X_cf, X])
            X = res_block(X, n_filters[i], kernel_init=kernel_init, kernel_reg=kernel_reg)
            # Prepare next deep features
            X_deep = X_skip
        
        # Classification
        class_act = 'sigmoid' if n_labels==1 else 'softmax'
        X = Conv2D(n_labels, (1, 1), padding='same', name='class_conv', kernel_initializer=kernel_init)(X)
        X = Activation(class_act, name='class_act')(X)
        
        # compile model
        model = Model(inputs=inputs, outputs=X)
    
    if to_compile:
        loss_fun = my_binary_loss if n_labels==1 else my_cat_loss
        acc_fun = 'accuracy' if n_labels==1 else 'categorical_accuracy'
        dice_fun = DiceCoef if n_labels==1 else cat_DiceCoef
        radam = tfa.optimizers.RectifiedAdam(lr=lr, beta_1=0.95, clipnorm=5.0)
        ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        model.compile(optimizer=ranger, loss=loss_fun, metrics=[acc_fun, dice_fun])
    
    return model


###############################################################################################################
# att-r2-unet
# https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-
###############################################################################################################
def se_block(input_feature, ratio=8, act='swish'):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """
    channel = input_feature.shape[-1]
    
    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    se_feature = Dense(channel // ratio,
                    activation=act,
                    kernel_initializer='he_normal',
                    use_bias=True,
                    bias_initializer='zeros')(se_feature)
    se_feature = Dense(channel,
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=True,
                    bias_initializer='zeros')(se_feature)
    
    se_feature = multiply([input_feature, se_feature])
    return se_feature


def cbam_block(cbam_feature, ratio=4, act='relu'):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio, act=act)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=4, act='relu'):
    channel = input_feature.shape[-1]
    
    shared_layer_one = Dense(channel//ratio,
                            activation=act,
                            kernel_initializer='he_normal',
                            use_bias=True,
                            bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                            kernel_initializer='he_normal',
                            use_bias=True,
                            bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7
    
    channel = input_feature.shape[-1]
    cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    
    return multiply([input_feature, cbam_feature])


def attention_up_and_concate(down_layer, layer, act='relu', kernel_reg=None):
    in_channel = down_layer.get_shape().as_list()[3]
    up = UpSampling2D(size=(2, 2))(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, act=act, kernel_reg=kernel_reg)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    out = my_concat([up, layer])
    return out


def attention_block_2d(x, g, inter_channel, act='relu', kernel_reg=None):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(g)
    f = Activation(act)(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])
    return att_x


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, kernel_size=[3, 3], stride=[1, 1], padding='same', act='relu', att_type=None, dropout=True, kernel_reg=None):
    input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(layer)
                layer1 = Activation(act)(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(add([layer1, layer]))
            layer1 = Activation(act)(layer1)
        layer = layer1
        layer = BatchNormalization()(layer)
    
    if dropout:
        # layer = Dropout(0.2)(layer)
        layer = SpatialDropout2D(0.2)(layer)
    
    if att_type == 'SE':
        layer = se_block(layer, act=act)
    elif att_type == 'CBAM':
        layer = cbam_block(layer, act=act)

    out_layer = add([layer, skip_layer])
    return out_layer


def res_btlnck(input_layer, n_filters, act='swish', kernel_reg=None):
    # res branch
    c_res = input_layer
    
    # bottleneck
    c1 = my_res_block(input_layer, c_res, n_filters, act, dr=1, kernel_reg=kernel_reg)
    c2 = my_res_block(c1, c_res, n_filters, act, dr=2, kernel_reg=kernel_reg)
    c3 = my_res_block(c2, c_res, n_filters, act, dr=4, kernel_reg=kernel_reg)
    c4 = my_res_block(c3, c_res, n_filters, act, dr=8, kernel_reg=kernel_reg)
    
    # res_add
    c5 = Add()([c1, c2, c3, c4])
    c5 = BatchNormalization()(c5)
    # c5 = Dropout(0.2)(c5)
    c5 = SpatialDropout2D(0.2)(c5)
    
    # res bottleneck
    if c5.shape[-1] != input_layer.shape[-1]:
        input_layer = Conv2D(c5.shape[-1], (1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(input_layer)
        input_layer = BatchNormalization()(input_layer)
    c = Add()([c5, input_layer])
    
    return c5


def my_res_block(input_layer, res_layer, n_filters, act, dr=1, kernel_reg=None):
    c = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same', dilation_rate=dr, kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(input_layer)
    c = BatchNormalization()(c)
    c = Activation(act)(c)
    if c.shape[-1] != res_layer.shape[-1]:
        res_layer = Conv2D(c.shape[-1], (1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(res_layer)
        res_layer = BatchNormalization()(res_layer)
    c = Add()([c, res_layer])
    return c


#Attention R2U-Net
def att_r2_unet(img_shape, n_labels=1, n_features=64, act='swish', btlneck=True, lr=3e-4, class_act=None, to_compile=True):
    # config
    depth = 3
    features = [n_features, 2*n_features, 6*n_features, 6*n_features]
    if class_act is None:
        class_act = 'sigmoid' if n_labels==1 else 'softmax'
    
    #kernel_reg = tf.keras.regularizers.l2(1e-3)
    kernel_reg = None
    
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    with strategy.scope():
    
        # inputs
        inputs = Input((img_shape[0], img_shape[1], img_shape[2]))
        x = inputs
        
        # encoder
        skips = []
        for i in range(depth):
            dropout = True if i > 0 else False
            x = rec_res_block(x, features[i], act=act, dropout=dropout, kernel_reg=kernel_reg)
            skips.append(x)
            x = MaxPooling2D((2, 2))(x)
        
        # botleneck
        if btlneck:
            x = res_btlnck(x, features[depth], act=act, kernel_reg=kernel_reg)
        else:
            x = rec_res_block(x, features[i], act=act)
        
        # decoder
        for i in reversed(range(depth)):
            dropout = True if i > 0 else False
            #att_type = None if i > 0 else 'CBAM'
            x = attention_up_and_concate(x, skips.pop(i), act=act, kernel_reg=kernel_reg)
            x = rec_res_block(x, features[i], act=act, dropout=dropout, kernel_reg=kernel_reg)
        
        # prediction
        conv6 = Conv2D(n_labels, (1, 1), padding='same')(x)
        conv7 = Activation(class_act)(conv6)
        
        # compile model
        model = Model(inputs=inputs, outputs=conv7)
        
        if to_compile:
            loss_fun = my_binary_loss if n_labels==1 else my_cat_loss
            acc_fun = 'accuracy' if n_labels==1 else 'categorical_accuracy'
            dice_fun = DiceCoef if n_labels==1 else cat_DiceCoef
            radam = tfa.optimizers.RectifiedAdam(lr=lr, beta_1=0.95, clipnorm=1.0)
            ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
            model.compile(optimizer=ranger, loss=loss_fun, metrics=[acc_fun, dice_fun])
    
    return model


################################################################################################################################################
# Custom losses
#
#
################################################################################################################################################
def DiceCoef(targets, inputs):
    return dice_coef(targets, inputs)


def cat_DiceCoef(targets, inputs, smooth=1e-6, channels_to_get=2):
    inputs = K.flatten(inputs[...,channels_to_get])
    targets = K.flatten(targets[...,channels_to_get])
    intersection = K.sum(targets * inputs)
    return (2.*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)


def my_binary_loss(targets, inputs, a=1., b=1.):
    # Dice loss
    dice_loss = a * dice(targets, inputs)
    
    # focal loss
    f_loss = b * focal_tversky(targets, inputs, alpha=0.7, gamma=4/3)

    return dice_loss + f_loss


def my_cat_loss(targets, inputs, smooth=1e-6, a=1.):
    # categorical focal loss
    f_loss = categorical_focal_loss_fixed(targets, inputs, alpha=[0.4, 0.6, 0.6])
    
    # Dice loss
    inputs = K.flatten(inputs[...,1:])
    targets = K.flatten(targets[...,1:])
    intersection = K.sum(targets * inputs)
    d_loss = 1 - (2.*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    
    return a*(d_loss + f_loss)


def categorical_focal_loss_fixed(y_true, y_pred, alpha=0.25, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)

    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(K.sum(loss, axis=-1))
