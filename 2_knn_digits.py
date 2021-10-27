import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from os import listdir


def imgDeal(file):
    '''
    处理文件
    将32x32矩阵转换为1x1024矩阵，方便计算距离
    '''
    newMat = np.zeros((1, 1024))
    fb = open(file)
    for i in range(32):
        lineStr = fb.readline()
        for j in range(32):
            newMat[0, 32*i+j] = int(lineStr[j])
    return newMat


def classify(x_data, y_data, labels, k):
    """
    kNN分类器
    x_data：训练集
    y_data:测试集
    labels:分类标签
    k：选取的分类区域

    欧氏距离：(d=(x-y)*2)*0.5
    """
    # 返回训练集行数
    xDataSize = y_data.shape[0]
    # 复制测试集，减去训练集
    diffMat = np.tile(x_data, (xDataSize, 1))-y_data
    # 求欧氏距离
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)  # 将矩阵每一行向量相加（平方求和）
    distance = sqDistance**0.5
    # 排序
    sortedDistance = distance.argsort()
    # 类别: 次数 的字典
    classified = {}
    for i in range(k):
        # 前k个元素的类别
        votedLabels = labels[sortedDistance[i]]
        # 计算类别次数
        # 返回votedLabels的值，若值不在字典中，返回默认值
        classified[votedLabels] = classified.get(votedLabels, 0)+1
    # 按出现频率降序排列
    sortedCounts = sorted(classified.items(),
                          key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的类别
    return sortedCounts[0][0]


def dataset():
    labels = []
    train_data = listdir('./machinelearning/Ch02/trainingDigits')
    m_train = len(train_data)  # 获取文件个数
    # 生成一个行=m，列=1024的零矩阵
    trainMat = np.zeros((m_train, 1024))
    for i in range(m_train):
        fileNameStr = train_data[i]
        fileStr = fileNameStr.split('.')[0]
        # 切割出真实标签
        fileNum = int(fileStr.split('_')[0])
        labels.append(fileNum)
        # 将训练集数据传入trainMat中
        trainMat[i, :] = imgDeal(
            './machinelearning/Ch02/trainingDigits/%s' % fileNameStr)
    errorCount = 0.0
    # 处理测试集
    test_data = listdir('./machinelearning/Ch02/testDigits')
    m_test = len(test_data)  # 获取文件个数
    for i in range(m_test):
        fileNameStr1 = test_data[i]
        fileStr1 = fileNameStr1.split('.')[0]
        # 切割出真实标签
        fileNum1 = int(fileStr1.split('_')[0])
        testMat = imgDeal(
            './machinelearning/Ch02/testDigits/%s' % fileNameStr1)
        classifyResult = classify(testMat, trainMat, labels, 3)
        print('预测结果：%s\t真实结果：%s' % (classifyResult, fileNum1))
        if(classifyResult != fileNum1):
            errorCount += 1
        errorPercent = errorCount/float(m_test)
    print("错误次数：%d\t总数：%d\t错误率：%f" % (errorCount, m_test, errorPercent))
    return None


if __name__ == "__main__":
    dataset()
