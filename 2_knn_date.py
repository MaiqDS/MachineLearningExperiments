import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from decimal import Decimal
# 从文本中解析数据

# 分类
# inX: 用于分类的输入向量
# dataSet: 训练样本集
# labels: 标签向量
# k: 选择最近邻居的数目

# def classify0(inX, dataSet, labels, k):
#     dataSetSize = dataSet.shape[0]  # 返回数据集行数(矩阵第二维长度)
#     # 在列方向上重复inX一次（横向），行方向上重复inX共dataSetSize次（纵向）
#     diffMat = tile(inX, (dataSetSize, 1))-dataSet
#     sqDiffMat = diffMat**2  # 二位特征相减后平方
#     sqDistances = sqDiffMat.sum(axis=1)
#     distances = sqDistances**0.5
#     sortedDistIndicies = distances.argsort()
#     classCount = {}
#     for i in range(k):
#         voteIlabel = labels[sortedDistIndicies[i]]
#         classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
#     sortedClassCount = sorted(
#         classCount.items(), key=operator.itemgetter(1), reverse=True)
#     return sortedClassCount[0][0]


def dataset():
    """
    打开并解析文件，对数据进行分类:
    1不喜欢
    2魅力一般
    3极具魅力
    """
    f = open("./machinelearning/Ch02/datingTestSet.txt")
    context = f.readlines()
    lineNum = len(context)  # 文件行数
    charaMat = np.zeros((lineNum, 3))  # 生成矩阵
    labels = []  # 标签向量
    index = 0
    for line in context:
        line = line.strip()  # 删除空白符
        everyLine = line.split('\t')  # 用tab切片
        charaMat[index, :] = everyLine[0:3]  # 提取前三列放入特征矩阵
        if everyLine[-1] == 'didntLike':
            labels.append(1)
        elif everyLine[-1] == 'smallDoses':
            labels.append(2)
        elif everyLine[-1] == 'largeDoses':
            labels.append(3)
        index += 1
    return charaMat, labels


def showdatas(mat, label, index):
    '''
    可视化
    '''
    fig = plt.figure()
    colors = []
    didntLike = mlines.Line2D([], [], color='black',
                              marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D(
        [], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D(
        [], [], color='red', marker='.', markersize=6, label='largeDoses')
    for i in labels:
        if i == 1:
            colors.append('black')
        elif i == 2:
            colors.append('orange')
        elif i == 3:
            colors.append('red')
    if index == 1:
        # 视频游戏与飞机里程数占比关系
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(x=mat[:, 0], y=mat[:, 1], color=colors, s=15)
        ax1.set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
        ax1.set_xlabel('每年获得的飞行常客里程数')
        ax1.set_ylabel('玩视频游戏所消耗时间占比')
        # 添加图例
        ax1.legend(handles=[didntLike, smallDoses, largeDoses])
        plt.show()
    elif index == 2:
        # 视频游戏与冰激凌之间的关系
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(x=mat[:, 1], y=mat[:, 2], color=colors, s=15)
        ax2.set_title('视频游戏消耗时间与每周消费的冰激凌公升数')
        ax2.set_xlabel('玩视频游戏消耗时间')
        ax2.set_ylabel('每周消费的冰激凌公升数')
        # 添加图例
        ax2.legend(handles=[didntLike, smallDoses, largeDoses])
        plt.show()
        # print(colors)

    elif index == 3:
        # 飞机里程数与冰激凌公升数的关系
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(x=mat[:, 0], y=mat[:, 2], color=colors, s=15)
        ax3.set_title('每年飞机飞行里程数与每周消费的冰激凌公升数')
        ax3.set_xlabel('每年获得的飞行常客里程数')
        ax3.set_ylabel('每周消费的冰激凌公升数')
        # 添加图例
        ax3.legend(handles=[didntLike, smallDoses, largeDoses])
        plt.show()

    return None


def autoNorm(x):
    '''
    归一化
    newValue = (oldValue - min) / (max - min)
    '''
    minvals = x.min(0)
    maxvals = x.max(0)
    ranges = maxvals-minvals
    # 建立与x相同结构的矩阵
    normx = np.zeros(np.shape(x))
    # 返回x的行数
    m = x.shape[0]
    # 原始值-最小值
    normx = x-np.tile(minvals, (m, 1))  # 将最小矩阵在行方向上复制1遍，列方向上m遍
    # 除以最大和最小值的差
    normx = normx/np.tile(ranges, (m, 1))
    return normx, ranges, minvals


def classify(x_data, y_data, labels, k):
    """
    kNN分类器
    x_data：测试集
    y_data:训练集
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


def classifyDataset(normx, labels):
    '''
    划分测试集(10%)与训练集(90%)
    '''
    # alpha = 0.1
    # # 获得归一化后数据集的行数
    # m = normx.shape[0]
    # # 划分测试集
    # numTest = int(m*alpha)
    # # 错误计数
    # errorCount = 0.0

    # for i in range(numTest):
    #     # 前numTest: 测试集，后m-numTest: 训练集
    #     classifyResult = classify(
    #         normx[i, :], normx[numTest:m, :], labels[numTest:m], 4)
    #     print("分类结果：%d\t真实类别: %d" % (classifyResult, labels[i]))
    #     if(classifyResult != labels[i]):
    #         errorCount += 1
    # errorPercent = errorCount/float(numTest)
    # print("错误次数: %d 总数: %d\t错误率：%f" % (errorCount, numTest, errorPercent))
    # return None

    # # k值与错误率对应关系字典
    # k_error = {}
    # k = 1
    # while k <= 900:
    #     errorCount = 0
    #     # 分类
    #     for i in range(numTest):
    #         # 前numTest: 测试集，后m-numTest: 训练集
    #         classifyResult = classify(
    #             normx[i, :], normx[numTest:m, :], labels[numTest:m], k)
    #         # print("分类结果：%d\t真实类别: %d" % (classifyResult, labels[i]))
    #         if(classifyResult != labels[i]):
    #             errorCount += 1
    #     errorPercent = errorCount/float(numTest)
    #     # print("错误次数: %d 总数: %d\t错误率：%f" % (errorCount, numTest, errorPercent))
    #     k_error[k] = errorPercent
    #     k += 1
    # return k_error

    # 获得归一化后数据集的行数
    m = normx.shape[0]
    alpha_error = {}
    alpha = 0.05
    step = 0.05
    while alpha < 1:
        # 划分测试集
        numTest = int(m*alpha)
        # 错误计数
        errorCount = 0.0
        for i in range(numTest):
            # 前numTest: 测试集，后m-numTest: 训练集
            classifyResult = classify(
                normx[i, :], normx[numTest:m, :], labels[numTest:m], 4)
            # print("分类结果：%d\t真实类别: %d" % (classifyResult, labels[i]))
            if(classifyResult != labels[i]):
                errorCount += 1
        errorPercent = errorCount/float(numTest)
        print("错误次数: %d 总数: %d\t错误率：%f" % (errorCount, numTest, errorPercent))
        alpha_error[alpha] = errorPercent
        alpha = float(Decimal(str(alpha))+Decimal(str(step)))
    print(alpha_error)
    return None


if __name__ == "__main__":
    charaMat, labels = dataset()
    # print(charaMat)
    # print(labels)
    showdatas(charaMat, labels, 3)
    # normx, ranges, minval = autoNorm(charaMat)
    # print(normx)
    # spe_error = classifyDataset(normx, labels)
    # print(k_error)
    # classifyDataset(normx, labels)
    # fig = plt.figure()
    # plt.title('k与准确率的关系')
    # plt.xlabel('k')
    # plt.ylabel('错误率')
    # plt.plot(k_error.keys(), k_error.values())
    # plt.show()
