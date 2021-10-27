import numpy as np
import random
import matplotlib.pyplot as plt


def loadDataset():
    '''引入数据'''
    dataMat = []
    labelMat = []
    fr = open("machinelearning\Ch06\\testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


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


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    dataMatIn: 数据集
    classLabels: 类别标签
    C: 常数C
    容错率: toler
    取消前最大的循环次数: maxIter
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while(iter < maxIter):  # iter：没有任何alpha改变的情况下遍历数据集的次数
        alphaPairsChanged = 0  # 记录alpha是否已优化
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T *
                        (dataMatrix*dataMatrix[i, :].T))+b  # 预测类别
            Ei = fXi-float(labelMat[i])  # Ei：误差
            # 若alpha可以更改，进入优化过程
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 随机选择第二个alpha值
                fXj = float(np.multiply(alphas, labelMat).T *
                            (dataMatrix*dataMatrix[j, :].T))+b
                Ej = fXj-float(labelMat[j])
                alphaIold = alphas[i].copy()  # 防止列表被改变，看不到新旧值变化
                alphaJold = alphas[j].copy()
                # 保证alpha在0与C之间
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L == H:  # 不做任何改变
                    print("L==H")
                    continue
                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T -\
                    dataMatrix[i, :]*dataMatrix[i, :].T -\
                    dataMatrix[j, :]*dataMatrix[j, :].T  # eta：alpha[j]的最优修改量
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):  # 未发生轻微改变
                    print("j not moving enough")
                    continue
                # 以相同修改量修改i、j
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # 修改方向相反
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - \
                    labelMat[j]*(alphas[j]-alphaJold) * \
                    dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - \
                    labelMat[j]*(alphas[j]-alphaJold) * \
                    dataMatrix[j, :]*dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" %
                      (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("========="+str(iter)+"==========")
    return b, alphas


def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(
        alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


# 启发式SMO算法的支持函数
class optStruct:
    '''新建一个类的收据结构，保存当前重要的值'''

    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


def calcEk(oS, k):
    '''格式化计算误差的函数，方便多次调用'''
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T *
                (oS.X*oS.X[k, :].T)+oS.b)
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
        classLabels).transpose(), C, toler)
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
                print("fullSet iter: %d i:%d,pairs changed %d" %
                      (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 统计alphas向量中满足0<alpha<C的alpha列表
            nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound iter: %d i:%d, pairs changed %d" %
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
    if((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
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
        eta = 2.0*oS.X[i, :]*oS.X[j, :].T-oS.X[i, :]*oS.X[i, :].T -\
            oS.X[j, :]*oS.X[j, :].T
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
        b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold) *\
            oS.X[i, :]*oS.X[i, :].T -\
            oS.labelMat[j]*(oS.alphas[j]-alphaJold) *\
            oS.X[i, :]*oS.X[j, :].T
        b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold) *\
            oS.X[i, :]*oS.X[j, :].T -\
            oS.labelMat[j]*(oS.alphas[j]-alphaJold) *\
            oS.X[j, :]*oS.X[j, :].T
        if(0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif(0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        # 如果有alpha对更新
        return 1
        # 否则返回0
    else:
        return 0


def drawPlot(dataMat, w, b):
    '''绘制样本点'''
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelArr[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(
        data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(
        data_minus_np)[1])
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7,
                        linewidth=1.5, edgecolor='red')
    plt.show()


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


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


if __name__ == "__main__":
    dataArr, labelArr = loadDataset()
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    print("Convergence.")
    # w = get_w(dataArr, labelArr, alphas)
    w = calcWs(alphas, dataArr, labelArr)
    # drawPlot(dataArr, w, b)
    showClassifer(dataArr, labelArr, w, b)
