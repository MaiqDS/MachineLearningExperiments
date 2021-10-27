import numpy as np
import random


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def selectJrand(i, m):
    '''
    在某区间内随机选择一个整数。
    i: 第一个alpha的下标
    m：所有alpha数目
    '''
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    '''
    调整大于H或小于L的alpha值。
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 启发式SMO算法的支持函数
class optStruct:
    '''新建一个类的收据结构，保存当前重要的值'''

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    '''格式化计算误差的函数，方便多次调用'''
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T*oS.K[:, k] + oS.b)
    Ek = fXk-float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    '''修改选择第二个变量alphaj的方法'''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 将误差矩阵每一行第一列置1，以此确定出误差不为0的样本
    oS.eCache[i] = [1, Ei]
    # 获取缓存中Ei不为0的样本对应的alpha列表
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    # 在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 否则，就从样本集中随机选取alphaj
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    '''更新误差矩阵'''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    '''SMO外循环'''
    # 保存关键数据
    oS = optStruct(np.mat(dataMatIn), np.mat(
        classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 选取第一个变量alpha的三种情况，从间隔边界上选取或者整个数据集
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # 没有alpha更新对
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet iter: %d i: %d,pairs changed: %d" %
                      (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 统计alphas向量中满足0<alpha<C的alpha列表
            nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound iter: %d i: %d, pairs changed: %d" %
                      (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        # 如果本次循环没有改变的alpha对，将entireSet置为true，
        # 下个循环仍遍历数据集
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def innerL(i, oS):
    '''内循环寻找alphaj'''
    # 计算误差
    Ei = calcEk(oS, i)
    # 违背kkt条件
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 计算上下界
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j]-oS.alphas[i])
            H = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 计算两个alpha值
        eta = 2.0*oS.K[i, j]-oS.K[i, i]-oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if(abs(oS.alphas[j]-alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i] *\
            (alphaJold-oS.alphas[j])
        updateEk(oS, i)
        # 在这两个alpha值情况下，计算对应的b值
        # 注，非线性可分情况，将所有内积项替换为核函数K[i,j]
        b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, i] - \
            oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i, j]
        b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, j] - \
            oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        # 如果有alpha对更新
        return 1
        # 否则返回0
    else:
        return 0


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def kernelTrans(X, A, kTup):  # calc the kernel or transform data to a higher dimensional space
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # linear kernel
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        # divide in NumPy is element-wise not matrix like Matlab
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("Kernel is not recognized")
    return K


def testDigits(kTup=('rbf', 10)):
    # 训练集
    dataArr, labelArr = loadImages(
        'machinelearning\Ch06\digits\\trainingDigits')
    k1 = 1.3
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T*np.multiply(labelSV, alphas[svInd])+b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    # 测试集
    dataArr, labelArr = loadImages('machinelearning\Ch06\digits\\testDigits')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    errorCount = 0
    m, n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T*np.multiply(labelSV, alphas[svInd])+b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))


if __name__ == "__main__":
    testDigits()
