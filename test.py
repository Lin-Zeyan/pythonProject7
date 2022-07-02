from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from keras.models import Model
from keras.utils import np_utils, plot_model

"""
数据集获取
"""


def get_mnist_data():
    # x_train_original和y_train_original代表训练集的图像与标签, x_test_original与y_test_original代表测试集的图像与标签
    (x_train_original, y_train_original), (x_test_original, y_test_original) = mnist.load_data()

    # 从训练集的60000张图像中，分离出10000张图像用作验证集
    x_val = x_train_original[50000:]
    y_val = y_train_original[50000:]
    x_train = x_train_original[:50000]
    y_train = y_train_original[:50000]
    # 原始数据量可视化
    print('训练集图像的尺寸：', x_train_original.shape)
    print('训练集标签的尺寸：', y_train_original.shape)

    # 将图像转换为四维矩阵(nums,rows,cols,channels), 这里把数据从unint8类型转化为float32类型, 提高训练精度。
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test_original.reshape(x_test_original.shape[0], 28, 28, 1).astype('float32')

    # 原始图像的像素灰度值为0-255，为了提高模型的训练精度，通常将数值归一化映射到0-1。
    x_train = x_train / 255
    x_val = x_val / 255
    x_test = x_test / 255


    # 图像标签一共有10个类别即0-9，这里将其转化为独热编码（One-hot）向量
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    y_test = np_utils.to_categorical(y_test_original)


    return x_train, y_train, x_val, y_val, x_test, y_test


"""
定义LeNet-5网络模型
LeNet网络由卷积层-池化层-卷积层-池化层-平铺层-全连接层-全连接层组成。
最后一层的全连接层采用softmax激活函数做10分类。
"""


def LeNet5():
    input_shape = Input(shape=(28, 28, 1))
    # C1层卷积层。输入特征图尺寸（输入图片）：32×32，卷积核尺寸：5×5，卷积核个数：6
    # 输出特征图尺寸 N = 28，卷积后的图像尺寸为28×28，因此每个输出特征图有28×28个神经元
    x = Conv2D(6, (5, 5), activation="relu", padding="same")(input_shape)
    # S2层池化层。输入特征图尺寸：28×28，池化尺寸：2×2，池化步长：2
    # 输出特征图尺寸N=14，通过池化区域为2、步长为2的最大池化层，可以对输入特征图实现降维，降维后的特征图尺寸是输入特征图尺寸的1/4
    # 卷积后的图像尺寸为14×14，因此每个输出特征图有14×14个神经元。
    x = MaxPooling2D((2, 2), 2)(x)
    # C3层卷积层。输入特征图尺寸：14×14，卷积核尺寸：5×5，卷积核个数：16
    # 输出特征图尺寸N=10，卷积后的图像尺寸为28×28，因此每个输出特征图有28×28个神经元。
    x = Conv2D(16, (5, 5), activation="relu", padding='same')(x)
    # S4层池化层。输入特征图尺寸：10×10，池化尺寸：2×2，池化步长：2
    # 输出图特征尺寸N=5，通过池化区域为2、步长为2的最大池化层，可以对输入特征图实现降维，降维后的特征图尺寸是输入特征图尺寸的1/4。
    # 卷积后的图像尺寸为5×5，因此每个输出特征图有5×5个神经元
    x = MaxPooling2D((2, 2), 2)(x)
    # 平铺层，降维，返回一个一维函数
    x = Flatten()(x)
    # F6层全连接层，输入维度：120，本层unit数：84，该层有84个特征图，特征图大小与C5一样都是1×1，与C5层全连接，激活函数为relu，输出形状为(None, 120, 120)
    x = Dense(120, activation='relu')(x)
    # 全连接层，激活函数为relu, 输出形状为(None, 84)
    x = Dense(84, activation='relu')(x)
    # 全连接层，激活函数为softmax，用于多分类的情形，输出形状为(None, 10)
    x = Dense(10, activation='softmax')(x)
    model = Model(input_shape, x)
    # 使用print(model.summary())得到神经网路的每一层参数
    print(model.summary())
    return model


"""
编译网络并训练
"""
x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data()
model = LeNet5()
# 编译网络（定义损失函数、优化器、评估指标）
# 使用交叉熵categorical_crossentropy作为损失函数，这是最常用的多分类任务的损失函数，常搭配softmax激活函数使用
# 优化器使用自适应矩估计：Adam
# 评价指标我们使用精度：accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 开始网络训练（定义训练数据与验证数据、定义训练代数，定义训练批大小）
# 使用model.fit()训练网络模型，函数中可以定义训练集数据与训练集标签，验证集数据与验证集标签、训练批次、批处理大小等
train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=32, verbose=2)
# 模型保存，.h5模型中包含模型结构、模型权重、模型编译信息等
model.save('lenet_mnist.h5')


# 定义训练过程可视化函数（训练集损失、验证集损失、训练集精度、验证集精度）
# 通过图的方式展示神经网路在训练时的损失与精度的变化
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

# 输出网络在测试集上的损失与精度
# Keras通过函数model.evaluate()测试神经网络在测试集上的情况
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 测试集结果预测
# 通过model.predict()对测试集图像进行预测
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
print('前20张图片预测结果：', predictions[:20])

# 预测结果图像可视化
(x_train_original, y_train_original), (x_test_original, y_test_original) = mnist.load_data()


# 多张图像可视化
def mnist_visualize_multiple_predict(start, end, length, width):
    for i in range(start, end):
        plt.subplot(length, width, 1 + i)
        plt.imshow(x_test_original[i], cmap=plt.get_cmap('gray'))
        title_true = 'true=' + str(y_test_original[i])
        # title_prediction = ',' + 'prediction' + str(model.predict_classes(np.expand_dims(x_test[i], axis=0)))
        title_prediction = ',' + 'prediction' + str(predictions[i])
        title = title_true + title_prediction
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show()


mnist_visualize_multiple_predict(start=0, end=9, length=3, width=3)

# 混淆矩阵
cm = confusion_matrix(y_test_original, predictions)
cm = pd.DataFrame(cm)
#构造一个类名
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 混淆矩阵可视化
def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, cmap='Oranges', linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


plot_confusion_matrix(cm)
