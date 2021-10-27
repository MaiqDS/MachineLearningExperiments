import matplotlib.pyplot as plt
import numpy as np
import random
from os import listdir

# 解析文本相关数据,提取每个特征组成向量,添加到数据矩阵
# 添加样本标签到标签向量


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# 从样本集中采取随机选择的方法选取第二个不等于第一个alphai的alphaj
# 优化向量alphaj


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# 约束范围L<alphaj<=H内更新后的alphaj的值


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def showDataSet(dataMat, labelMat):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()

# 算法思想
# 先创建一个alpha向量，并将起初始化为0


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    :param dataMatIn: 数据列表
    :param classLabels: 标签列表
    :param C: 权衡因子（增加松弛因子，变成了软间隔问题，会在目标优化时使用）
    :param toler:容错率
    :param maxIter:最大的迭代次数
    :return:
    '''
    dataMatrix = np.mat(dataMatIn)
    # 将输入数据列表转化为矩阵形式
    labelMat = np.mat(classLabels).transpose()
    # 将标签列表也转化为矩阵，并将其转置方便后续做乘法处理
    b = 0
    # 初始化b=0
    m, n = np.shape(dataMatrix)
    # 获取样本数据的条数与特征数，条数：m，特征数：n
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化一个alpha的0向量
    iter_num = 0
    # 初始迭代次数为0
    while (iter_num < maxIter):
        alphaPairsChanged = 0
        # alphaPairsChanged标识着alpha被改变的对数
        for i in range(m):
            # 遍历所有样本
            fXi = float(np.multiply(alphas, labelMat).T *
                        (dataMatrix * dataMatrix[i, :].T)) + b
            # 计算支持向量机的算法预测
            Ei = fXi - float(labelMat[i])
            # 计算预测与实际的误差
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 通过两个方向,去检查误差门限,并确保an不为上下限
                j = selectJrand(i, m)
                # 随机选取第二个变量alphaj
                fXj = float(np.multiply(alphas, labelMat).T *
                            (dataMatrix * dataMatrix[j, :].T)) + b
                # 计算第二个变量的预测值
                Ej = fXj - float(labelMat[j])
                # 计算alphaj的误差
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 记录alphai和alphaj的原始值,方便后续进行比较
                if (labelMat[i] != labelMat[j]):
                    # 如果两个alpha对应的标签不一样,则计算相应的上下界
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    # 在同一侧
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # -----------计算aj是否满足可行域-----------
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i,
                                                                               :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                # 根据公式计算,这里的eta可以看成去度量两个样本i和j的相似性,这里的地方需要用核函数,来取代上面的内积,这里直接用xi当作了核函数
                if eta >= 0:
                    # 如果eta>=0代表在这个范围内,不能当作一个支持向量,直接返回
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 更新aj的值,使得目标函数最大化
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 查看更新的aj值是否在边界中,如果不在,说明优化的值跑出了边界L和H,那么对aj的值进行裁剪,回收到这个范围
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    # 修正的步长过于小,导致无法迅速的找到最优解,没有太多意义,直接返回
                    print("alpha_j变化太小")
                    continue
                alphas[i] += labelMat[j] * \
                    labelMat[i] * (alphaJold - alphas[j])
                # 选好优化的aj后,对ai进行优化
                # -----------计算阈值b--------------
                # 作用使得对两个样本i和j都满足kkt条件,这样是我们smo算法,选出最优的退出条件
                # 使得所有样本都满足ktt条件
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 如果ai不在边界上,则b=b1,同里,如果都在边界上,那么取阈值的平均值
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("第%d次迭代 样本:%d, alpha优化次数:%d" %
                      (iter_num, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b, alphas

# 只针对于二维问题


def showClassifer(dataMat, labelMat, w, b):
    # 绘制样本点
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
    # 解决中文乱码
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(
        data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(
        data_minus_np)[1], s=30, alpha=0.7)
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1_down, y2_down = (1-b - a1 * x1) / a2, (1-b - a1 * x2) / a2
    y1_up, y2_up = (-1-b-a1 * x1) / a2, (-1-b-a1 * x2) / a2
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2], color='red', label='决策面')
    plt.plot([x1, x2], [y1_up, y2_up], linestyle="--",
             color='green', label='间隔面')
    plt.plot([x1, x2], [y1_down, y2_down], linestyle="--", color='green')
    plt.legend(loc='lower right')
    for i, alpha in enumerate(alphas):
        # enumerate()
        # 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，
        if abs(alpha) > 0:
            # 只有alpha大于0才有意义
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7,
                        linewidth=1.5, edgecolor='red')
    plt.show()


def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(
        alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()

# -------------完整的SMO算法-------------------
# 启发式SMO算法的支持函数
# 一个类的收据结构,保存当前的值


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        # 初始化结构体,同时将这些值进行初始化
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        # 将特征用核函数处理一遍
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


# 核转化函数
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    k = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        # 如果核函数类型为lin类型 则就是直接相乘就行
        k = X*A.T
    elif kTup[0] == "rbf":
        # 如果核函数类型为rbf:径向基核函数
        # 将每个样本向量利用核函数转为高维空间
        for j in range(m):
            deltaRow = X[j, :]-A
            k[j] = deltaRow*deltaRow.T
        k = np.exp(k/(-1*kTup[1]**2))
    elif kTup[0] == 'poly':  # 多项式核
        k = X * A.T
        for j in range(m):
            k[j] = k[j] ** kTup[1]
    elif kTup[0] == 'laplace':  # 拉普拉斯核
        for j in range(m):
            deltaRow = X[j, :] - A
            k[j] = deltaRow * deltaRow.T
            k[j] = np.sqrt(k[j])
        k = np.exp(-k / kTup[1])
    else:
        raise NameError(
            'Houston we Have a problem , That kenerl is not recognised')
    return k


# 误差计算
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 选择aj使误差的变化最大


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0
        eta = 2.0*oS.K[i, j]-oS.K[i, i]-oS.K[j, j]
        if eta >= 0:
            # print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print("alpha_j变化太小")
            return 0
        oS.alphas[i] += oS.labelMat[j] * \
            oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
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
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kernelOption=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(
        classLabels).transpose(), C, toler, kernelOption)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        # print("迭代次数: %d" % iter)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

# -------------基于手写识别问题----------------


def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

 # 将图像转为向量


def img2vector(filename):
    featVec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            featVec[0, 32 * i + j] = int(lineStr[j])
    return featVec


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages(
        'machinelearning\Ch06\digits\\trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    # print("基核参数的取值为:%d",kTup[1])
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    errorCount = (float(errorCount) / m) * 100
    print("the training error rate is: %.2f%%" % errorCount)

    dataArr, labelArr = loadImages('machinelearning\Ch06\digits\\testDigits')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    errorCount = (float(errorCount) / m)*100
    print("the test error rate is: %.2f%%" % errorCount)


# --------------病马预测------------------
# 文件解析函数,将文件数据转化为特征矩阵,标签矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 20))
    classLabelVector = []  # 标签矩阵
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去除文本文件中的回车符'\n'
        listFromLine = line.split('\t')  # 根据tab符进行划分,返回的是列表
        returnMat[index, :] = listFromLine[0:20]
        x = int(float(listFromLine[-1]))
        if x == 1:
            classLabelVector.append(1)
        elif x == 0:
            classLabelVector.append(-1)
        # classLabelVector.append(int(float(listFromLine[-1])))
        index += 1
    return returnMat, classLabelVector

# 将数据进行归一化处理


def autoNorm(dataSet):  # 归一化处理
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet


def colicTest(kTup=('rbf', 10)):
    trainingSet, traingLabels = loadImages(
        'machinelearning\Ch06\digits\\trainingDigits')
    trainingSet = autoNorm(trainingSet)
    b, alphas = smoP(trainingSet, traingLabels, 200, 0.001, 100, kTup)
    datMat = np.mat(trainingSet)
    labelMat = np.mat(traingLabels).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("基核参数的取值为:%d" % kTup[1])
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(traingLabels[i]):
            errorCount += 1
    errorCount = (float(errorCount) / m) * 100
    print("the training error rate is: %.2f%%" % errorCount)

    testSet, testLabels = loadImages(
        'machinelearning\Ch06\digits\\testDigits')
    testSet = autoNorm(testSet)
    errorCount = 0
    datMat = np.mat(testSet)
    labelMat = np.mat(testLabels).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(testLabels[i]):
            errorCount += 1
    errorCount = (float(errorCount) / m) * 100
    print("the test error rate is: %.2f%%" % errorCount)
    return errorCount


def multTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        print("第 %d 次 预测" % (k+1))
        # rbf,laplace,lin,poly
        errorSum += colicTest(('laplace', 10))
    print("after %d iterations the average error rate is : %f" %
          (numTests, errorSum/float(numTests)))


if __name__ == '__main__':
    # dataArr, classLabels = loadDataSet('testSet.txt')
    # b, alphas = smoSimple(dataArr, classLabels , 0.6, 0.001, 40)
    # w = get_w(dataArr, classLabels , alphas)
    # showClassifer(dataArr, classLabels , w, b)

    # b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40)
    # w = calcWs(alphas, dataArr, classLabels)
    # showClassifer(dataArr,classLabels, w, b)
    #

    testDigits(('rbf', 100))
    # colicTest(('poly',4))
    # colicTest(('rbf',10))
    # multTest()
    # 牛逼
