from math import log
import operator
import matplotlib.pyplot as plt

'''---------------------------------------------建立决策树-----------------------------------------------'''


def createDataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'],
               [3, 0, 'maybe']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def calcShannonEnt(dataset):
    '''计算给定数据集的香农熵'''
    num = len(dataset)  # 数据集样本数量
    labelCount = {}  # 创建数据字典，键：样本类别（最后一列），值：样本数量
    # 计算每种类别下的样本数量，并将其放在字典中对应的键下
    for featureVec in dataset:
        label = featureVec[-1]  # 最后一列
        if label not in labelCount.keys():
            labelCount[label] = 1
        else:
            labelCount[label] += 1
    # 计算数据集的熵
    shannonEnt = 0.0
    for key in labelCount.keys():
        prob = float(labelCount[key])/num  # 求概率p(xi)=本类别数/总数
        shannonEnt -= prob*log(prob, 2)  # H=-sigma(p(xi)*log_2(p(xi)))
    return shannonEnt


def splitDataset(dataset, feature, value):
    '''
    按照给定的特征划分数据集。
    dataSet: 待划分的数据集
    feature: 划分数据集的特征
    value: 特征值
    '''
    sDataset = []
    # 抽取符合特征的数据
    for featureVector in dataset:
        if featureVector[feature] == value:
            reducedFeature = featureVector[:feature]
            reducedFeature.extend(featureVector[feature+1:])
            sDataset.append(reducedFeature)
    return sDataset


def chooseBestFeatureToSplit(dataset):
    '''选择最好的数据集划分方式'''
    numFeatures = len(dataset[0])-1  # 求特征个数
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]  # 用列表推导式将第i个特征的值提取出来
        uniqueVals = set(featList)  # 利用集合的互异性找出特征的不同取值
        newEntropy = 0.0
        for value in uniqueVals:  # 当前特征的不同取值
            subDataset = splitDataset(dataset, i, value)  # 按不同特征划分数据集
            # 求新划分的数据集的香农熵
            prob = len(subDataset)/float(len(dataset))
            newEntropy += prob*calcShannonEnt(subDataset)
        infoGain = baseEntropy-newEntropy
        if(infoGain > bestInfoGain):  # 找到最佳的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''多数表决法定义叶子节点的分类'''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritem(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataset, labels):
    '''递归构建决策树'''
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):  # 所有类标签完全相同，直接返回该类标签
        return classList[0]
    if len(dataset[0]) == 1:  # 使用完所有特征，仍不能将数据集划分成仅包含唯一类别的分组，使用多数表决法决定叶子节点分类
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)  # 选择划分数据集最佳特征的索引
    bestFeatLabel = labels[bestFeat]  # 根据特征索引提取索引的名称
    decisionTree = {bestFeatLabel: {}}  # 将此特征作为树的根节点
    del labels[bestFeat]  # 将已放进树中的特征从特征标签中删除
    featValues = [example[bestFeat] for example in dataset]  # 提取所有样本关于这个特征的取值
    uniqueVals = set(featValues)  # 利用集合互异性，提取这个特征的不同取值
    for value in uniqueVals:  # 根据特征的不同取值，创建这个特征所对应节点的分支
        subLabels = labels[:]
        decisionTree[bestFeatLabel][value] = createTree(
            splitDataset(dataset, bestFeat, value), subLabels)
    return decisionTree


'''--------------------------------------------绘制决策树--------------------------------------------'''


def getNumLeafs(tree):
    '''获取叶节点的数目，在绘制决策树时确定x轴的长度'''
    numOfLeaf = 0
    firstNode, = tree.keys()
    secondDict = tree[firstNode]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            # 若子节点为字典，则该节点也是一个判断节点，递归调用
            numOfLeaf += getNumLeafs(secondDict[key])
        else:
            numOfLeaf += 1
    return numOfLeaf


def getTreeDepth(tree):
    '''计算树的深度，在绘制决策树时确定y轴的高度'''
    depthOfTree = 0
    firstNode, = tree.keys()  # 第一次划分数据集的类别标签
    secondDict = tree[firstNode]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            # 若子节点为字典，则该节点也是一个判断节点，递归调用
            thisNodeDepth = getTreeDepth(secondDict[key])+1
        else:  # 到达叶子节点
            thisNodeDepth = 1
        if thisNodeDepth > depthOfTree:
            depthOfTree = thisNodeDepth
    return depthOfTree


def plotNode(nodeTxt, nodeIndex, parentNodeIndex, nodeType):
    '''
    绘制节点
    nodeTxt: 文本内容
    nodeIndex: 文本中心点
    parentNodeIndex: 箭头指向文本的点
    nodeType: 点的类型

    annotate(text,xy,xycoords,xytext,textcoords,va,ha,bbox,arrowprops)
    xy: 进行标注的点的坐标
    xytext: 标注的文本信息的位置
    xycoords, textcoords: xy和xytext的说明，默认为data
    va，ha: 文本框中文字的位置，va表示竖直方向，ha表示水平方向
    bbox: 文字边框样式
    arrowprops: 箭头样式
    '''
    plt.annotate(nodeTxt, xy=parentNodeIndex, xycoords='axes fraction', xytext=nodeIndex,
                 textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=dict(arrowstyle='<-'))


def plotMidText(thisNodeIndex, parentNodeIndex, text):
    '''在父子节点之间添加注释'''
    xmid = (parentNodeIndex[0]-thisNodeIndex[0])/2.0+thisNodeIndex[0]
    ymid = (parentNodeIndex[1]-thisNodeIndex[1])/2.0+thisNodeIndex[1]
    plt.text(xmid, ymid, text)  # 在指定位置添加注释


def plotTree(tree, parentNodeIndex, midTxt):
    # 决策节点；设置文本框的类型和文本框背景灰度，范围为0-1，0为黑，1为白，不设置默认为蓝色
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')
    leafNode = dict(boxstyle='round4', fc='1')  # 设置叶子节点文本框的属性
    numOfLeafs = getNumLeafs(tree)
    nodeTxt, = tree.keys()
    nodeIndex = (plotTree.xOff+(1.0+float(numOfLeafs))/2.0 /
                 plotTree.totalW, plotTree.yOff)  # 计算节点的位置
    plotNode(nodeTxt, nodeIndex, parentNodeIndex, decisionNode)
    plotMidText(nodeIndex, parentNodeIndex, midTxt)  # 标记子节点属性
    # 减少y偏移
    secondDict = tree[nodeTxt]
    plotTree.yOff = plotTree.yOff-1.0/plotTree.totalD
    # 绘制节点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], nodeIndex, str(key))
        else:
            plotTree.xOff = plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,
                     plotTree.yOff), nodeIndex, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), nodeIndex, str(key))
    plotTree.yOff = plotTree.yOff+1.0/plotTree.totalD


def createPlot(tree):  # 绘制决策树的主函数
    # 创建一个画布，命名为'decisionTree',画布颜色为白色
    fig = plt.figure('DecisionTree', facecolor='white')
    fig.clf()  # 清空画布
    # 111：将画布分成1行1列，去第一块画布；frameon：是否绘制矩形坐标框
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    # xOff和yOff追踪已绘制节点的位置，计算放置下一个节点的恰当位置。
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(tree, (0.5, 1.0), '')
    plt.xticks([])
    plt.yticks([])
    plt.show()


'''------------------------------------------测试决策树------------------------------------------'''


def classify(inputTree, featureLabels, testVector):
    firstNode, = inputTree.keys()
    secondDict = inputTree[firstNode]
    featureIndex = featureLabels.index(firstNode)
    for key in secondDict.keys():
        if testVector[featureIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(
                    secondDict[key], featureLabels, testVector)
            else:  # 进入叶子节点
                classLabel = secondDict[key]
    return classLabel


'''------------------------------------------存储决策树------------------------------------------'''


def storeTree(inputTree, filename):
    import pickle
    file = open(filename, 'wb')
    pickle.dump(inputTree, file)
    file.close()


def loadTree(filename):
    import pickle
    file = open(filename, 'rb')
    tree = pickle.load(file)
    file.close()
    return tree


'''--------------------------------------------主函数--------------------------------------------'''

if __name__ == '__main__':
    myDat, labels = createDataset()
    decisionTree = createTree(myDat, labels)
    # storeTree(decisionTree, 'decisionTree')
    # myTree = loadTree('decisionTree')
    featureLabels = ['no surfacing', 'flippers']
    # createPlot(myTree)
    while(True):
        x = ""
        xlist = ""
        x = input()
        xlist = x.split(",")
        xlist = [int(xlist[i]) for i in range(len(xlist))]
        print(classify(decisionTree, featureLabels, xlist))
