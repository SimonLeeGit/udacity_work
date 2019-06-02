# 猫狗大战 (Dogs vs Cats)

## 项目概述

## 项目指南

1. 克隆存储库并进入到项目文件夹。

   ```
   git clone https://github.com/SimonLeeGit/udacity_work.git
   cd udacity_work/dogs_vs_cats/
   ```

2. (Optional) 默认所需数据集已经包含在项目中（datas目录下），如有需要可以从kaggle上下载。[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

3. 为数据集下载VGG-16关键特征，并将其放置于目录下。

4. 配置安装环境及相关的Python依赖包。

   对于 __Mac/OSX__：

   ```
   conda env create -f requirements/mac.yml
   conda activate dogs_vs_cats
   KERAS_BACKEND=tensorflow python -c "from keras import backend"
   ```

   对于 __Windows__：

   ```
   conda env create -f requirements/windows.yml
   conda activate dogs_vs_cats
   set KERAS_BACKEND=tensorflow
   python -c "from keras import backend"
   ```

5. 打开notebook

   ```
   jupyter notebook dogs_vs_cats.ipynb
   ```
