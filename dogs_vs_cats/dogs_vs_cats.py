
# coding: utf-8

# # 猫狗大战（Dogs vs Cats）
# 
# ## 项目介绍
# 
# **TODO：介绍Dogs vs Cats 项目的背景**
# 
# ## 项目内容
# 
# * [Step0](#step0): 导入数据集
# * [Step1](#step1): 数据分析
# * [Step2](#step2): 模型预测
# * [Step3](#step3): 结果分析
#  
# <a id="step0"></a>
# ## 1. 导入数据集
# 
# 读取datas目录下的数据集，并分为train和test两类。

# In[ ]:

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

data = load_files('datas')

files = np.array(data['filenames'])
targets = np.array(data['target'])
target_names = np.array(data['target_names'])

train_files = [files[idx] for idx in range(len(files)) if targets[idx] == np.argwhere(target_names == 'train')]
test_files = [files[idx] for idx in range(len(files)) if targets[idx] == np.argwhere(target_names == 'test')]

print("There are {} train images.".format(len(train_files)))
print("There are {} test images.".format(len(test_files)))


# 可视化训练数据集中前12张图片。

# In[ ]:

# import cv2
# import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

# def visualize_img(img_path, ax):
#     img = cv2.imread(img_path)
#     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
# fig = plt.figure(figsize=(20, 10))
# for i in range(12):
#     ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
#     visualize_img(train_files[i], ax)


# <a id="step1"></a>
# ## 2. 数据分析
# 
# ### 2.1 提取数据特征
# 
# 通过查看训练数据集，发现数据集的标注信息是定义在文件名的，从文件名中提取对应的数据特征，如下表：
# 
# |文件|标注（dog）|标注（cat）|
# |-|-|-|
# |datas/train/cat.6938.jpg  |0|1|
# |datas/train/dog.11432.jpg |1|0|
# |datas/train/cat.433.jpg   |0|1|
# |datas/train/cat.11305.jpg |0|1|
# 

# In[ ]:

data_labels = ("dog", "cat")

train_labels = []
for file in train_files:
    for idx in range(len(data_labels)):
        if data_labels[idx] in file:
            train_labels.append(idx)

train_targets = np_utils.to_categorical(np.array(train_labels), 2)
print("The first 5 train file:\n{}\n".format(train_files[0:5]))
print("The first 5 train targets:\n{}\n".format(train_targets[0:5]))


# ## 2.2 数据处理
# 
# 根据文件名，读取图片文件中的数据

# In[ ]:

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# In[ ]:

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# ### 2.2 验证集划分
# 
# 从sklearn.model_selection中导入train_test_split
# 将train_files和train_targets作为train_test_split的输入变量
# 设置test_size为0.2，划分出20%的验证集，80%的数据留作新的训练集。
# 设置random_state随机种子，以确保每一次运行都可以得到相同划分的结果。（随机种子固定，生成的随机序列就是确定的）

# In[ ]:

from sklearn.model_selection import train_test_split

X_train , X_valid , y_train, y_valid = train_test_split(train_tensors, train_targets, test_size=0.2, random_state=100)

print("Splited train set num: {}".format(len(X_train)))
print("Splited valid set num: {}".format(len(X_valid)))


# <a id="step2"></a>
# ## 3. 模型预测
# 
# 基于[Deep Residual Networks](https://github.com/KaimingHe/deep-residual-networks#models) 来创建CNN模型。
# 
# **TODO: 介绍为什么使用ResNet-50，以及ResNet模型的原理**
# 
# ### 3.1 搭建CNN模型
# 
# 使用Keras来构建CNN模型，详细的安装配置请参考README。

# In[ ]:

from res_util import ResNet

# Create ResNet
resnet = ResNet()

# Build ResNet50 Model
model = resnet.Build_ResNet50((3,224,224), 2)


# ## 3.2 训练模型
# 
# 使用构建好的ResNet50模型进行训练，并将训练的权重保存到hdf5文件中。

# In[ ]:

from keras.callbacks import ModelCheckpoint

# Trainning Model
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
        epochs=30, batch_size=32, callbacks=[checkpointer], verbose=1)


# 将模型的训练过程进行可视化。

# In[ ]:

# import matplotlib.pyplot as plt

# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()


# ## 3.3 验证模型
# 
# 根据测试集中的数据进行预测。

# In[ ]:

# Load trained model weights
model.load_weights('saved_models/weights.best.Resnet50.hdf5')

# Predict test set
test_predict = [np.argmax(model.predict(np.expand_dims(feature, axis=0), verbose=1)) for feature in test_tensors]


# 将部分预测的结果进行可视化预览

# In[ ]:

# plot a random sample of test images, their predicted labels
# fig = plt.figure(figsize=(20, 8))
# for i, idx in enumerate(np.random.choice(test_tensors.shape[0], size=32, replace=False)):
#     ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
#     ax.imshow(np.squeeze(test_tensors[idx]))
#     pred_idx = np.argmax(test_predict[idx])
#     ax.set_title("{}".format(data_labels[pred_idx]), color=("green"))


# 将预测的结果提交到kaggle，以便得到模型的准确率排名。

# In[ ]:

import pandas as pd

## 加载结果格式
submit_frame = pd.read_csv("submission/sample_submission.csv")
## 保存结果
submit_frame['label'] = test_predict
test_result_name = "submission/submission.csv"
submit_frame[['id','label']].to_csv(test_result_name,index=False)


# <a id="step3"></a>
# ## 4. 结果分析
# 
# ### 4.1 测试模型
# 
# **TODO: 分析算法的不足，提供优化建议，并提供参数优化后的结果**

# In[ ]:

# model.load_weights('saved_models/weights.best.Resnet50.hdf5')

# Resnet50_predict = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# print("The first 5 train targets:\n{}\n".format(train_targets[0:5]))


# ### 4.2 提交结果

# In[ ]:

# ## 加载结果格式
# submit_frame = pd.read_csv("sample_submission.csv")
# ## 保存结果
# result = pd.merge(submit_frame, test_content, on="id", how='left')
# result = result.rename(index=str, columns={"cuisine_y": "cuisine"})
# test_result_name = "tfidf_cuisine_test.csv"
# result[['id','cuisine']].to_csv(test_result_name,index=False)

