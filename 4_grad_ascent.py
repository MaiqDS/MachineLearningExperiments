import numpy as np
import matplotlib.pyplot as plt
import random


def loadDataset():
    dataMat = []
    labelMat = []
    fr = open(
        'E:\CDU\\2021-2022_1\人工智能\Experiments\machinelearning\Ch05\\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 为计算方便，将X0设为1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def plotBestFit(weights):
    dataMat, labelMat = loadDataset()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]  # 最佳拟合直线
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # matIn训练数据
    labelMat = np.mat(classLabels).transpose()  # 将行向量转为列向量
    m, n = np.shape(dataMatrix)  # m行，n列
    alpha = 0.001  # 学习率
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)  # 向量运算，h为列向量
        error = (labelMat-h)
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    '''随机梯度下降SGD1'''
    # 实际上不是随机的，只是每次只用一个样本计算梯度，minibatch大小为1
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    alpha = 0.01  # 学习率
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))  # 有新样本时，进行增量更新而非全部重新遍历
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights.tolist()


def stocGradAscent1(dataMatrix, classLabels, numIter=600):
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001  # 减小alpha，每次减小1/(j+i)
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights.tolist()


if __name__ == '__main__':
    dataArr, labelMat = loadDataset()
    # weight = gradAscent(dataArr, labelMat)
    # weight = stocGradAscent1(dataArr, labelMat)
    # plotBestFit(weight.getA())  # 将numpy矩阵转换为数组
    # plotBestFit(weight)
    plotBestFit(gradAscent(dataArr, labelMat).getA())
