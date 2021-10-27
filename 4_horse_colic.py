import numpy as np
import random
import matplotlib.pyplot as plt
import time


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


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


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights.tolist()


def classifyVector(inX, weights):
    '''分类算法'''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open(
        'machinelearning\Ch05\horseColicTraining.txt', encoding='utf8')
    frTest = open('machinelearning\Ch05\horseColicTest.txt', encoding='utf8')
    trainingSet = []
    trainingLabels = []
    # print(frTest)
    # print(frTrain)
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # trainWeights = stocGradAscent0(np.array(trainingSet), trainingLabels)
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 80)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        temp = k+1
        print("-------round %d------" % temp)
        start = time.perf_counter()
        errorSum += colicTest()
        end = time.perf_counter()
        print("the run time is: ", end-start)

    print("after %d iterations the average error rate is: %f" %
          (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    # start = time.perf_counter()
    multiTest()
    # colicTest()
    # end = time.perf_counter()
    # print("the run time is: ", end-start)
