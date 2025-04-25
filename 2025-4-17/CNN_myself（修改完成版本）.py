#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:18:23 2025

@author: UrBoge
"""

# 导包
import os
import copy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
# CNN模型需要的包
from Bio.Alphabet import IUPAC
from keras.optimizers import Adam
from contextlib import redirect_stdout
# 绘图相关
from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
    average_precision_score, confusion_matrix, f1_score, \
    matthews_corrcoef



# Step1.加载输入数据，处理后保存

'''
先通过HDM（高密度突变）技术生成抗体突变库，针对CDRH3区域。
    Ab+筛选：细胞与抗体特异性荧光染料孵育，流式细胞术检测高荧光（表达抗体）细胞。
    Ag+/Ag-筛选：Ab+细胞与荧光标记的HER2孵育，流式细胞术检测高荧光（Ag+）或低荧光（Ag-）。
突变库在细胞中表达，即为Ab+序列（表达抗体的细胞）；剩余的为Ab-序列，原因可能是引入了终止子，导致无法在现实合成，故丢弃。
一部分进行筛选，可以得出Ag-和Ag+ ；
另一部分未筛选的归为未结合（因为大多数都不会和抗原结合），即文件中的_Ab.txt 。
这一步首先需要加载的就是MiXCR（一种分析免疫组库数据的工具）的.txt文件，输出将是带有抗原分类标签的序列数据(结合为1，非结合为0)。
先定义两个方法，一个是mixcr_input方法，另一个是load_input_data方法。后者需要使用前者的定义。
mixcr_input方法可以读取MiXCR的标准.txt文件，提取出克隆计数，克隆比例，核苷酸序列，氨基酸序列和分类标签。去除包含终止密码子和不完整密码子的序列，得到一个5列的DataFrame
load_input_data方法可以将多个文件传入mixcr_input方法，然后整合到一起，删除重复行，删除氨基酸序列的前三个和后两个，得到长度为10的氨基酸序列，并且打乱数据重排。
'''

# 定义第一个函数，mixcr_input
def mixcr_input(file_name, Ag_class, seq_len):
    """
    从 MiXCR 文本输出文件中读取数据

    Parameters
    ---
    file_name: 要读取的 MiXCR 文本文件的文件名。

    Ag_class: 抗原分类标签（即，抗原结合物 = 1，非结合物 = 0）。

    seq_len: 期望的序列长度（氨基酸个数），不符合此长度的将被移除。
    """

    # 读取数据并重命名列
    
    x = pd.read_table(file_name)  # 使用pd.read_table读取指定路径的.txt文件（MiXCR标准输出），这一步会将pd DataFrame赋值给x，包含文件的所有列
    x = x[['Clone count', 'Clone fraction',
           'N. Seq. CDR3 ', 'AA. Seq.CDR3 ']]
    x = x.rename(index=str, columns={
        'Clone count': 'Count', 'Clone fraction': 'Fraction',
        'N. Seq. CDR3 ': 'NucSeq', 'AA. Seq.CDR3 ': 'AASeq'
    })  # 上述代码将从DataFrame中选择4列，并重命名为新的名称（原始输出有空格，重新命名后没有了），这样我们的x就是一个精简的，只包含4列的DataFrame。
        # Count：克隆计数  Fraction：克隆比例  NucSeq：CDR3核苷酸序列  AASeq：CDR3氨基酸序列

    # 选择长度并删除重复序列
    x = x[(x.AASeq.str.len() == seq_len) & (x.Count > 1)]  # seq_len是我们输入的希望的序列长度，即，只保留指定长度的序列。x.Count即为克隆计数大于1的序列，即，至少出现两次。
                                                           # 这样的目的是为了保持序列一致性；计数过滤是为了过滤低丰度序列，减少假阳性，提高数据的可读性。
    x = x.drop_duplicates(subset='AASeq')  # 根据AASeq列去除重复序列，如果多个克隆具有相同的氨基酸序列，只保留第一次出现的记录。
                                           # 避免重复序列对后续分析的干扰，DMS处理后的数据中可能包含重复克隆

    # 去除终止密码子和不完整的密码子序列（*，_），即：去除非功能性序列
    idx = [i for i, aa in enumerate(x['AASeq']) if '*' not in aa]  # 过滤包含终止密码子的序列，实现的过程是，检查AASeq中是否有*（终止子），idx是符合条件的行索引。
    x = x.iloc[idx, :]  # 保留这些行
    idx = [i for i, aa in enumerate(x['AASeq']) if '_' not in aa]  # 过滤包含不完整密码子的序列，实现的过程是，检查AASeq中是否有_（不完整密码子），idx是符合条件的行索引。
    x = x.iloc[idx, :]  # 保留这些行

    # 如果 Ag_class == 0，所有序列标记为非结合（负样本）。
    # 如果 Ag_class == 1，所有序列标记为结合（正样本）。
    if Ag_class == 0:
        x['AgClass'] = 0  # 为 DataFrame 添加一列AgClass，根据输入参数Ag_class赋值。
    if Ag_class == 1:
        x['AgClass'] = 1  # 为 DataFrame 添加一列AgClass，根据输入参数Ag_class赋值。

    return x  # 返回处理后的 DataFrame，包含5列，即Count（克隆计数）, Fraction（克隆比例）, NucSeq（核苷酸序列）, AASeq（氨基酸序列）, AgClass（抗原分类标签）

# 定义第二个函数，load_input_data
def load_input_data(filenames, Ag_class):
    """
    加载指定的文件名中的文件

    Parameters
    ---
    filenames: 一个指定要加载的文件名的列表

    Ag_class: 从 MiXCR 文本文件中的分类序列，整数类型，表示序列是否与抗原结合（抗原结合者 = 1，非结合者 = 0）
    """

    # 合并非结合序列数据集
    # 非结合数据集包括 Ab+ 数据和 Ag- 数据
    # 所有 3 个库的排序数据
    l_data = []  # 创建一个空列表，用于存储每个文件处理后的数据。
    for file in filenames:
        l_data.append(
            mixcr_input('data/' + file, Ag_class, seq_len=15)
        )

    """
    对于上述for循环解释如下：
    1.遍历filenames中的每个文件名
    2.对每个文件，调用mixcr_input函数（即上述方法1），将每一个输入的mixcr生成的标准.txt文件处理成有五列的pandas dataframe形式，具体解释为：
        （1）mixcr_input方法处理输入的txt文件，开始应当在终端使用cd指令进入主文件夹，指定代码文件夹（这里是scripts）的具体代码文件。
            所以传入的第一个参数是"data"+file，这里的file是遍历的filenames，这个是方法二的传入参数，即文件名组成的列表。
        （2）将mixcr_input方法处理后的数据追加（append）至先前创建的l_data列表中。
        （3）从后续代码可以看出，这个Ag_class的值是自己分类的，即分类为0/1是先确定的，然后根据分类将文件进行读取。
    """
    mHER_H3 = pd.concat(l_data) # 默认按行拼接，即数据合并到一起。

    # 删除重复序列
    mHER_H3 = mHER_H3.drop_duplicates(subset='AASeq') # 确保每个氨基酸序列只出现一次，保留第一次出现的记录。
                                                      # 不过这里的克隆计算和克隆比例不一定一致，但是本文献提出的模型在训练时并没有考虑。

    # 去除 'CAR/CSR' 基序和最后两个氨基酸，保留长度为10的氨基酸序列（因为前面定义了seq_len=15）。
    # （存疑）另外这里根据GROK的回答，开头可能是抗体框架区的保守序列，结尾是固定序列（J基因或其他区域）
    # （强调）这里应当注意开头都是CAR和CSR，否则可能会错误的删掉开头的序列，是需要交流探讨的问题，应该注意
    mHER_H3['AASeq'] = [x[3:-2] for x in mHER_H3['AASeq']]

    # 打乱序列并重置索引
    mHER_H3 = mHER_H3.sample(frac=1).reset_index(drop=True)  # 这里的.sample(frac=1)表示将所有的数据打乱，后面的.reset_index就是丢弃原本的行标签。

    return mHER_H3  # 最终得到的就是包括5列的，AAseq长度为10。

# 下面使用上述定义的函数来加载序列，需要注意的是：
"""
未结合序列的文件命名均为： “_Ab.txt” 或 “_AgN.txt”
    _Ab.txt文件的数据均为Ab+（抗体表达群体），即成功表达抗体的细胞，但未测试抗原结合。经过进一步询问AI，应该指的是：现实中可以设计出现的？？（存疑）
    _AgN.txt文件的数据均为Ag-（抗原非结合群体），即明确不与抗原结合的群体。这些序列来自流式细胞术筛选，细胞与抗原（HER2）孵育后，未显示荧光信号（即未结合抗原）。
结合序列的文件命名均为： “_2Ag647.txt” 或 “_2Ag488.txt”
    两者表示经过两轮抗原特异性富集（Ag+1 和 Ag+2）的序列，使用 647 和 488 荧光标记（流式细胞术筛选），即明确与抗原结合的群体。
"""

# 加载未结合序列
ab_neg_files = [
    'mHER_H3_1_Ab.txt', 'mHER_H3_1_AgN.txt',
    'mHER_H3_2_Ab.txt', 'mHER_H3_2_AgN.txt',
    'mHER_H3_3_Ab.txt', 'mHER_H3_3_AgN.txt'
]
mHER_H3_AgNeg = load_input_data(ab_neg_files, Ag_class=0)  # 类别标签：结合为1，非结合为0

# 加载结合序列
ab_pos_files = [
    'mHER_H3_1_2Ag647.txt', 'mHER_H3_1_2Ag488.txt',
    'mHER_H3_2_2Ag647.txt', 'mHER_H3_2_2Ag488.txt',
    'mHER_H3_3_2Ag647.txt', 'mHER_H3_3_2Ag488.txt'
]
mHER_H3_AgPos = load_input_data(ab_pos_files, Ag_class=1)  # 类别标签：结合为1，非结合为0

# 保存文件
mHER_H3_AgNeg.to_csv('data/mHER_H3_AgNeg.csv')
mHER_H3_AgPos.to_csv('data/mHER_H3_AgPos.csv')

# 这一步结束，最后的变量为：mHER_H3_AgNeg ，mHER_H3_AgPos。一个是所有的正样本集，一个是所有的负样本集。

# Step2.数据预处理
# 定义第三个函数，data_split_adj
from sklearn.model_selection import train_test_split
def data_split_adj(Ag_pos, Ag_neg, fraction):
    """
    创建一个数据集的集合，并将其拆分为训练集和两个测试集。数据集经过调整，以匹配指定的类别分割比例，该比例决定了Ag+序列的比例。

    Parameters
    ---
    Ag_pos: 包含正样本（Ag+，抗原结合序列）的 DataFrame
    Ag_neg: 包含负样本（Ag-，非结合序列）的 DataFrame
    fraction: 期望的正样本在组合数据集中的比例（0-1之间）
    """

    # 定义一个类对象，每次实例化的时候都会将输入的键值对直接存储为对象属性
    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    # 计算目标数据量（根据传入参数fraction，即期望的正样本在组合数据集中的比例）,这一步是为了下一步做铺垫。
    data_size_pos = len(Ag_pos)/fraction  # 满足比例所需要的组合数据集大小，因为len(df)输出的就是df的行数，后续不再赘述。
    data_size_neg = len(Ag_neg)/(1-fraction)  # 满足比例所需要的组合数据集大小

    # 调整正负样本数量，以满足期望的正样本所占比例
    # 下面的代码是基于正样本和负样本数量 及 上一步得到的满足比例所需要的组合数据集大小 data_size_pos 和 data_size_neg 进行的：
   
    if len(Ag_pos) <= len(Ag_neg):  # 这下面就是正样本数量不多于负样本数量的情况
        if data_size_neg < data_size_pos:  # 比较上一步的理论组合数据集大小，若正集计算出的大于负集，说明负样本集限制了总数据量，则按照小的理论组合数据集大小来进行。
            Ag_pos1 = Ag_pos[0:int((data_size_neg*fraction))]  # 正样本按照期望比例来取
            # 下一行是自己加的，等同于下下行代码的语法，是为了理解计算过程，也是为了解释部分可能出现的疑惑。
            # Ag_neg1 = Ag_neg[0:int((data_size_neg*(1-fraction)))]  # 负样本也按照期望比例来取
            Ag_neg1 = Ag_neg  # 上一行注释等同于所有的负样本都进入组合数据集
            Unused = Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)]  # 将未使用的正样本放入Unused中

        if data_size_neg >= data_size_pos:  # 同理，使用较小的那个理论组合数据集大小
            # 下一行是自己加的，等同于下下行代码的语法，是为了理解计算过程，也是为了解释部分可能出现的疑惑。
            # Ag_pos1 = Ag_pos  # 下一行代码的逻辑等同于这个，是为了理解计算过程，也是为了解释部分可能出现的疑惑。
            Ag_pos1 = Ag_pos[0:int((data_size_pos*(fraction)))] # 正样本按照期望比例来取，其实就是所有的正样本集都进入组合数据集
            Ag_neg1 = Ag_neg[0:int((data_size_pos*(1-fraction)))]  # 负样本按照期望比例来取
            Unused = pd.concat(
                [Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_pos*(1-fraction))):len(Ag_neg)]]
            )  # 将未使用的正负样本放入Unused中

    else:  # 这下面就是正样本数量多于负样本数量的情况，计算同上，不做赘述。
        if data_size_pos < data_size_neg:
            Ag_pos1 = Ag_pos
            Ag_neg1 = Ag_neg[0:(int(data_size_pos*(1-fraction)))]
            Unused = Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)]

        if data_size_pos >= data_size_neg:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*(fraction)))]
            Ag_neg1 = Ag_neg[0:int((data_size_neg*(1-fraction)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_neg*(1-fraction))):len(Ag_neg)]]
            )

    # 将调整后的正负样本进行合并，命名为Ag_combined，进行去重、重新排序操作。
    Ag_combined = pd.concat([Ag_pos1, Ag_neg1])  # 合并，默认按行拼接
    Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')  # 去除重复序列
    Ag_combined = Ag_combined.sample(frac=1).reset_index(drop=True)  # 打乱顺序，重置索引

    # 按照70%训练集，30%测试集进行分割。
    idx = np.arange(0, Ag_combined.shape[0])  # 这里的 Ag_combined.shape[0] 和 len(Ag_combined.shape) 结果应当是一致的。
    idx_train, idx_test = train_test_split(
        idx, stratify=Ag_combined['AgClass'], test_size=0.3
    )  # 为了保证AgClass列的每个类别在训练集和测试集中都按相同比例分布。
       # 使用 stratify=Ag_combined['AgClass'] ，保证idx_train 和 idx_test 的比例与AgClass一致。

    # 对测试集再次进行50%分割，即最后得到的是15%验证集和15%测试集。
    idx2 = np.arange(0, idx_test.shape[0])  
    idx_val, idx_test2 = train_test_split(
        idx2, stratify=Ag_combined.iloc[idx_test, :]['AgClass'], test_size=0.5
    )  # 因为是对测试集进行的划分，所以这里使用的是idx_test对应的索引，仔细辨别即可~

    # 创建并返回结果
    Seq_Ag_data = Collection(
        train=Ag_combined.iloc[idx_train, :],  # 训练集是上面分类出来的行索引，对应起来就是以 DataFrame 作为 value 值。
        val=Ag_combined.iloc[idx_test, :].iloc[idx_val, :],  # 测试集2是通过两次步骤得到的，所以第一次先索引到测试集1，然后再把idx_val提取得出。
        test=Ag_combined.iloc[idx_test, :].iloc[idx_test2, :],  # 同上方验证集，先索引到测试集1，然后再把idx_test2提取得出
        complete=Ag_combined  # 这个就是完整的数据集。
    )

    return Seq_Ag_data, Unused

# 创建包含训练集和测试集划分的集合
# 根据上面对函数的定义，mHER_all_adj 就是包含训练集（70%），验证集（15%），测试集（15%），完整数据集的字典；unused_seq就是没有使用的序列。
mHER_all_adj, unused_seq = data_split_adj( 
    mHER_H3_AgPos, mHER_H3_AgNeg, fraction=0.5
)

# 创建数据集合的浅拷贝
mHER_all_copy = copy.copy(mHER_all_adj)

# 创建目录以存储图形（硬编码，后续可能会改）
os.makedirs('figures', exist_ok=True)  # 使用 os.makedirs 创建一个名为 figures 的目录



# Step3.定义并运行分类模型（CNN）
from Bio.Alphabet import IUPAC
from keras.optimizers import Adam
from contextlib import redirect_stdout

# 定义第四、五个函数，one_hot_encoder 和 one_hot_decoder，前者用于编码，后者用于解码。
def one_hot_encoder(s,  alphabet):
    """
    将一个生物序列编码成为one-hot编码矩阵

    Parameters
    ---
    s: 字符串格式，待编码的生物序列
    alphabet: BioPython 的 Alphabet 对象，定义序列的字母表（vocabulary），即20种标准氨基酸。
              下载地址：http://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC-module.html

    Example
    ---
    sequence = 'CARGSSYSSFAYW'
    one_hot_encoder(s=sequence, alphabet=IUPAC.protein)

    Returns
    ---
    x: array, n_size_alphabet, n_length_string
        Sequence as one-hot encoding
    """

    # 构造编码的字典
    d = {a: i for i, a in enumerate(alphabet.letters)}

    # Encode
    x = np.zeros((len(d), len(s)))  # 创建一个全零矩阵
    x[[d[c] for c in s], range(len(s))] = 1  # 对序列s的每一个字符c，在对应列的索引d[c]处置1，形成独热编码。

    return x

def one_hot_decoder(x, alphabet):
    """
    Decodes a one-hot encoding to a biological sequence

    Parameters
    ---
    x: array, n_size_alphabet, n_length_string
        Sequence as one-hot encoding
    alphabet: Alphabet object, downloaded from
        http://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC-module.html

    Example
    ---
    encoding = one_hot_encoder(sequence, IUPAC.unambiguous_dna)
    one_hot_decoder(encoding, IUPAC.unambiguous_dna)

    Returns
    ---
    s : str, decoded sequence
    """

    d = {a: i for i, a in enumerate(alphabet.letters)}
    inv_d = {i: a for a, i in d.items()}
    s = (''.join(str(inv_d[i]) for i in np.argmax(x, axis=0)))

    return s


# 定义第六个函数，create_cnn
def create_cnn(units_per_layer, input_shape,
               activation, regularizer):
    """
    create_cnn是一个基于Keras构建生成卷积神经网络（CNN）模型的函数。

    Parameters
    ---
    units_per_layer: 列表，定义 CNN 每一层的类型和参数。每个元素是一个子列表，表示一层，格式为：
        卷积层。Filter information: [CONV, # filters, kernel size, stride]
        最大池化层。Max Pool information: [POOL, pool size, stride]
        Dropout层。Dropout information: [DROP, dropout rate]
        展平层。Flatten: [FLAT]
        全连接层。Dense layer: [DENSE, number nodes]

    input_shape：元组，定义输入数据的形状

    activation： 激活函数，字符串或 Keras 激活对象。例如，'relu'

    regularizer: 正则化器，应用于卷积层和全连接层的权重和偏置。例如，keras.regularizers.l1(0.01)。论文中未使用正则化（None）。
    """

    # 初始化 CNN 模型
    model = keras.Sequential()  # 创建一个Keras Sequential模型。

    # 输入层
    model.add(keras.layers.InputLayer(input_shape))  # 添加输入层，指定输入数据的形状。这里形状应当是(10, 20)

    # 构建网络，这一层通过遍历 units_per_layer，再根据每层的类型和参数动态添加层。
    for i, units in enumerate(units_per_layer):  # 这里的i值为迭代次数，units是参数一中的每一个列表元素（列表中的元素均为列表）。
        if units[0] == 'CONV':  # 如果是卷积层，添加1D卷积层，提取序列的局部特征。
            model.add(keras.layers.Conv1D(filters=units[1],  # 前面参数定义时的第二个就是：滤波器数量
                                          kernel_size=units[2],  # 卷积核大小
                                          strides=units[3],  # 步幅
                                          activation=activation,  # 激活函数
                                          kernel_regularizer=regularizer,  # 正则化器
                                          bias_regularizer=regularizer,  # 正则化器
                                          padding='same'))  # 通过填充保持输出长度与输入一致
        elif units[0] == 'POOL':  # 如果是池化层，添加1D最大池化层，降维并提取重要特征。
            model.add(keras.layers.MaxPool1D(pool_size=units[1],  # 池化窗口大小
                                             strides=units[2]))  # 步幅
        elif units[0] == 'DENSE':  # 如果是全连接层，添加全连接层，整合特征进行分类。
            model.add(keras.layers.Dense(units=units[1],  # 神经元数量，论文是50
                                         activation=activation,  # 激活函数，例如："relu"
                                         kernel_regularizer=regularizer,  # 正则化
                                         bias_regularizer=regularizer))  # 正则化
        elif units[0] == 'DROP':  # 如果是DROPOUT列表，添加Dropout层，随机丢弃部分神经元，防止过拟合。
            model.add(keras.layers.Dropout(rate=units[1]))  # 丢弃比例
        elif units[0] == 'FLAT':  # 如果是展平列表，就展平。
            model.add(keras.layers.Flatten())  # 展平
        else:
            raise NotImplementedError('Layer type not implemented')  # 确保函数鲁棒性，防止无效参数

    # 输出层，激活函数为: Sigmoid
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model

# 定义第七、八个函数，plot_ROC_curve 和 plot_PR_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
    average_precision_score, confusion_matrix, f1_score, \
    matthews_corrcoef
import matplotlib as plt
def plot_ROC_curve(y_test, y_score, plot_title, plot_dir):
    """
    绘制ROC特征曲线，用于评估分类器的性能。
    
    Parameters
    ---
    y_test: 真实标签，二分类标签（0或1），0表示不结合HER2,1表示结合HER2
    y_score: 分类器预测的概率值（0-1之间）
    plot_title: 图表的标题
    plot_dir: 保存图像的路径
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_score)  # 使用sklearn.metrics.roc_curve 计算 ROC 曲线的假阳性率（FPR）、真阳性率（TPR）和阈值。
    roc_auc = auc(fpr, tpr)  # 使用 sklearn.metrics.auc 计算 ROC 曲线下面积（Area Under the Curve, AUC）。
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area - %0.2f)' % roc_auc)  # 以FPR为x轴，TPR为y轴，绘制曲线，颜色暗橙，线宽为2，图例标签显示AUC值（保留两位小数）
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 绘制对角线（y=x），表示随机分类器的 ROC 曲线（AUC=0.5）。
    plt.xlim([0.0, 1.0])  # 设置x轴范围
    plt.ylim([0.0, 1.05])  # 设置y轴范围
    plt.xlabel('False Positive Rate')  # x轴标注为“假阳性率”，FPR
    plt.ylabel('True Positive Rate')   # y轴标注为“真阳性率”，TPR
    plt.title(plot_title)  # 设置图表标题
    plt.legend(loc='lower right')  # 添加图例，显示 ROC 曲线和 AUC 值。
    savefig(plot_dir)  # 保存图表到指定路径。
    plt.cla()  # 清除当前轴（cla）释放内存。
    plt.clf()  # 清除当前图形（clf），释放内存。

def plot_PR_curve(y_test, y_score, plot_title, plot_dir):
    """
    绘制 PR 曲线并保存到指定目录。

    Parameters
    ---
    y_test: 真实序列标签
    y_score: 预测概率
    plot_title: 标题
    plot_dir: 保存路径
    """

    precision, recall, thresholds = precision_recall_curve(y_test, y_score)  # 计算 PR 曲线的精确度、召回率和阈值
    average_precision = average_precision_score(y_test, y_score)  # 使用 sklearn.metrics.average_precision_score 计算平均精确度（AP）。

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})  # 检查 plt.fill_between 是否支持 step 参数，设置填充样式。
    plt.step(recall, precision, color='navy', alpha=0.2, where='post',
             label='Avg. Precision: {0:0.2f}'.format(average_precision))  # 绘制 PR 曲线（阶梯图）。
    plt.fill_between(recall, precision, alpha=0.2, color='navy', **step_kwargs)  # 以 Recall 为 x 轴，Precision 为 y 轴，绘制阶梯曲线。
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(plot_title)
    plt.legend(loc='lower right')
    savefig(plot_dir)
    plt.cla()
    plt.clf()


# 定义第九个函数，calc_stat
def calc_stat(y_test, y_pred):
    """
    计算分类器的性能指标，通过混淆矩阵计算准确率、精确度、召回率，并计算F1分数和MCC相关系数

    Parameters
    ---
    y_test: 真实标签，0或1
    y_pred: 预测标签

    Returns
    ---
    stats: NumPy数组，形状 (5,)，包含 [accuracy, precision, recall, f1, mcc]
    """

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred
    ).ravel()  # 使用 sklearn.metrics.confusion_matrix 计算混淆矩阵，提取真阴性（TN）、假阳性（FP）、假阴性（FN）和真阳性（TP）。
    
    acc = (tp+tn)/(tp+tn+fp+fn)  # 计算准确率（Accuracy），即正确预测的比例。
    prec = (tp)/(tp+fp)  # 计算精确度（Precision），即预测为正的样本中真正正的比例。
    recall = tp/(tp+fn)  # 计算召回率（Recall），即真正正样本中被正确预测的比例。

    # Calculate F1 and MCC score
    f1 = f1_score(y_test, y_pred)  # 使用 sklearn.metrics.f1_score 计算 F1 分数，即精确度和召回率的调和平均数。
    mcc = matthews_corrcoef(y_test, y_pred)  # 计算 Matthews 相关系数，综合考虑所有混淆矩阵元素。

    # Return statistics
    return np.array([acc, prec, recall, f1, mcc])  # 返回一个numpy数组，结果就是这五个指标。

# 定义第十个函数，CNN_classification
def CNN_classification(dataset, filename, save_model=False, params=None):
    """
    使用卷积神经网络对数据进行分类，随后绘制ROC和PR曲线。

    参数
    ---
    dataset: 输入数据集，包含训练和测试分割数据，以及绑定和非绑定序列的相应标签（1和0）。

    filename: 一个标识符，用于区分不同的图表。

    save_model：可选；如果提供，应指定保存模型概要和权重的目录。在这种情况下，将返回分类模型。
                如果为False，则返回包含分类准确率、精确度和召回率的数组。

    params: 可选；如果提供，应指定在单独模型调优步骤中确定的优化模型参数。如果为None，则模型参数为硬编码。
    """

    # 导入训练/测试集（选择序列作为特征："AASeq"）
    X_train = dataset.train.loc[:, 'AASeq'].values
    X_test = dataset.test.loc[:, 'AASeq'].values
    X_val = dataset.val.loc[:, 'AASeq'].values

    # 对序列进行独热编码
    X_train = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_train]
    X_train = np.transpose(np.asarray(X_train), (0, 2, 1))
    X_test = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_test]
    X_test = np.transpose(np.asarray(X_test), (0, 2, 1))
    X_val = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_val]
    X_val = np.transpose(np.asarray(X_val), (0, 2, 1))

    # 提取训练/测试/验证集的标签（"AgClass"即为标签）
    y_train = dataset.train.loc[:, 'AgClass'].values
    y_test = dataset.test.loc[:, 'AgClass'].values
    y_val = dataset.val.loc[:, 'AgClass'].values

    # 设置CNN的参数
    if not params:
        params = [['CONV', 400, 3, 1],
                  ['DROP', 0.5],
                  ['POOL', 2, 1],
                  ['FLAT'],
                  ['DENSE', 50]]

    # 使用上述指定的参数创建CNN
    CNN_classifier = create_cnn(params, (10, 20), 'relu', None)

    # 编译CNN：配置模型的优化器、损失函数和评估指标。
    opt = Adam(learning_rate=0.000075)  # Adam 优化器，学习率为0.000075
    CNN_classifier.compile(optimizer=opt, loss='binary_crossentropy',  # 二元交叉熵损失，适合二分类任务
                           metrics=['accuracy'])  # 训练时监控准确率

    # 将CNN拟合到训练集
    _ = CNN_classifier.fit(
        x=X_train, y=y_train, shuffle=True, validation_data=(X_val, y_val),  # shuffle=True：每轮训练前打乱数据。
        epochs=20, batch_size=16, verbose=2  # 训练20轮，每批16个样本，显示简洁的训练日志。
    )

    # 预测测试集结果
    y_pred = CNN_classifier.predict(x=X_test)  # 使用训练好的模型预测测试集的结合概率。

    # ROC曲线
    title = 'CNN ROC曲线 (训练集={})'.format(filename)
    plot_ROC_curve(
        y_test, y_pred, plot_title=title,
        plot_dir='figures/CNN_ROC_Test_{}.png'.format(filename)
    )

    # 精确度-召回率曲线
    title = 'CNN 精确度-召回率曲线 (训练集={})'.format(filename)
    plot_PR_curve(
        y_test, y_pred, plot_title=title,
        plot_dir='figures/CNN_P-R_Test_{}.png'.format(filename)
    )
    # 如果指定，则保存模型
    if save_model:
        # 模型概要
        with open(os.path.join(save_model, 'CNN_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                CNN_classifier.summary()

        # 模型权重
        CNN_classifier.save(
            os.path.join(save_model, 'CNN_HER2')
        )

        # 返回分类模型
        return CNN_classifier
    else:
        # 大于0.5的概率被认为是显著的
        y_pred_stand = (y_pred > 0.5)

        # 计算统计数据
        stats = calc_stat(y_test, y_pred_stand)

        # 返回统计数据
        return stats



# 下面的代码（Step3内），所有的多行索引都是源代码，下面的内容即修改代码。（只运行CNN模型）

"""
# 为最终 DataFrame 创建列名称
ML_columns = ('Train_size', 'LogReg_acc', 'LogReg_prec', 'LogReg_recall',
              'LogReg_F1', 'LogReg_MCC', 'LogReg2D_acc', 'LogReg2D_prec',
              'LogReg2D_recall', 'LogReg2D_F1', 'LogReg2D_MCC',
              'KNN_acc', 'KNN_prec', 'KNN_recall', 'KNN_F1', 'KNN_MCC',
              'LSVM_acc', 'LSVM_prec', 'LSVM_recall', 'LSVM_F1', 'LSVM_MCC',
              'SVM_acc', 'SVM_prec', 'SVM_recall', 'SVM_F1', 'SVM_MCC',
              'RF_acc', 'RF_prec', 'RF_recall', 'RF_F1', 'RF_MCC',
              'ANN_acc', 'ANN_prec', 'ANN_recall', 'ANN_F1', 'ANN_MCC',
              'CNN_acc', 'CNN_prec', 'CNN_recall', 'CNN_F1', 'CNN_MCC',
              'RNN_acc', 'RNN_prec', 'RNN_recall', 'RNN_F1', 'RNN_MCC')
ML_df = pd.DataFrame(columns=ML_columns)
"""
# 原始代码替换成下面的内容（后续的更改不再进行注释提示）：
# 为最终 DataFrame 创建列名称，仅包含 CNN 指标和 Train_size
ML_columns = ('Train_size', 'CNN_acc', 'CNN_prec', 'CNN_recall', 'CNN_F1', 'CNN_MCC')
ML_df = pd.DataFrame(columns=ML_columns)

# 将未使用的序列添加到训练集，并且同时也准备好测试集和验证集，然后在后面作为参数传入。
for x in np.linspace(0, 10000, 11): # 从0到10000创建11个等间隔点，这里的输出结果为[0，1000，2000，...，10000]
    x = int(x)  # 将x转换为整数，可用于索引

    # 将 x 个未使用的序列添加到训练集
    mHER_all_copy.train = pd.concat(
        [copy.copy(mHER_all_adj.train), unused_seq[0:x]]
    )

    # 打乱训练数据，这里只有训练集添加了unused里面的数据内容，依次添加0，1000，...，10000
    mHER_all_copy.train = mHER_all_copy.train.sample(
        frac=1
    ).reset_index(drop=True)
    
    mHER_all_copy.test = copy.copy(mHER_all_adj.test)
    mHER_all_copy.val = copy.copy(mHER_all_adj.val)

    '''
    # 运行所有分类器
    LogReg_stats = LogReg_classification(
        mHER_all_copy, '{}'.format(x)
    )
    LogReg2D_stats = LogReg2D_classification(
        mHER_all_copy, '{}'.format(x)
    )
    KNN_stats = KNN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    LSVM_stats = LSVM_classification(
        mHER_all_copy, '{}'.format(x)
    )
    SVM_stats = SVM_classification(
        mHER_all_copy, '{}'.format(x)
    )
    RF_stats = RF_classification(
        mHER_all_copy, '{}'.format(x)
    )
    ANN_stats = ANN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    CNN_stats = CNN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    RNN_stats = RNN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    '''
    # 仅运行 CNN 分类器
    CNN_stats = CNN_classification(
        mHER_all_copy, '{}'.format(x)
    )


    # 从下面的多行注释可以注意到，如果三个引号在最左边，相当于前面的循环就终止了，这样的话会导致下面的all_stats缩进错误，进而报错，因此应当缩进4行。
    """
    # 添加一行包含所有统计信息的数据
    all_stats = np.concatenate(
        (np.array([x]), LogReg_stats, LogReg2D_stats, KNN_stats, LSVM_stats,
         SVM_stats, RF_stats, ANN_stats, CNN_stats, RNN_stats)
    )
    ML_df = ML_df.append(
        pd.DataFrame([all_stats], columns=list(ML_columns)), ignore_index=True
    )

# 将统计信息保存到文件中
ML_df.to_csv('figures/ML_increase_negs_combined.csv')
    """

    # 添加一行包含 CNN 统计信息的数据
    all_stats = np.concatenate(
        (np.array([x]), CNN_stats)
    )


    ML_df = ML_df.append(pd.DataFrame([all_stats], columns=list(ML_columns)), ignore_index=True)

    '''
    # 这里面的代码是上面代码的升级版本，但是目前还是pandas早期版本，所以先用上面的。
    ML_df = pd.concat(
        [ML_df, pd.DataFrame([all_stats], columns=list(ML_columns))], 
        ignore_index=True
    )
    '''
# 将统计信息保存到文件中
ML_df.to_csv('figures/CNN_increase_negs_combined.csv')



# Step4.在计算机生成的数据上运行CNN分类器

# 定义第十一个函数，data_split
def data_split(Ag_pos, Ag_neg):
    """
    将抗原结合（Ag+）和非结合（Ag-）的抗体序列数据组合，并去除重复序列。
    创建一个特殊的测试集，强制正样本（Ag+）占比约为 10%，负样本（Ag-）占比约为 90%，以模拟不平衡场景（正样本稀少）。

    Parameters
    ---
    Ag_pos: DataFrame，包含抗原结合（Ag+）的抗体序列，AgClass=1
    Ag_neg: DataFrame，包含非结合（Ag-）的抗体序列，AgClass=0

    Returns
    ---
    Seq_Ag_data: 一个 Collection 对象，包含了训练集，验证集，测试集和完整的合并数据集。
    """

    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    # 合并正负样本，并且去重，打乱重设索引。
    Ag_combined = pd.concat([Ag_pos, Ag_neg])
    Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_combined = Ag_combined.sample(frac=1).reset_index(drop=True)

    # 70%训练集和30%测试集，保持类分布一致。
    idx = np.arange(0, Ag_combined.shape[0])
    idx_train, idx_test = train_test_split(
        idx, stratify=Ag_combined['AgClass'], test_size=0.3
    )

    # 测试集进一步分出15%测试集和15%验证集
    idx2 = np.arange(0, idx_test.shape[0])
    idx_val, idx_test2 = train_test_split(
        idx2, stratify=Ag_combined.iloc[idx_test, :]['AgClass'], test_size=0.5
    )

    # 调整测试集，使负样本占 90%，正样本占 10%。
    Test_AgNeg_amt = len(Ag_combined.iloc[idx_test2, :]['AgClass']) - \
        Ag_combined.iloc[idx_test, :].iloc[idx_test2, :]['AgClass'].sum()  # 负样本数 = 总行数 - 正样本数
    Adj_test_amt = int(Test_AgNeg_amt/0.90)  # 负样本应占 90%，因此总行数 = 负样本数 / 0.9。
    Test_AgPos_amt = Adj_test_amt - Test_AgNeg_amt  # 正样本数 = 总行数 - 负样本数。
    Test_AgPos = Ag_combined.iloc[idx_test, :].iloc[idx_test2, :].iloc[np.where(
        Ag_combined.iloc[idx_test, :].iloc[idx_test2, :]['AgClass'] == 1)[0], :]
    Test_AgNeg = Ag_combined.iloc[idx_test, :].iloc[idx_test2, :].iloc[np.where(
        Ag_combined.iloc[idx_test, :].iloc[idx_test2, :]['AgClass'] == 0)[0], :]
    Updated_test = pd.concat([Test_AgNeg, Test_AgPos[0:Test_AgPos_amt]])

    # 创建Collection
    Seq_Ag_data = Collection(train=Ag_combined.iloc[idx_train, :],
                             val=Ag_combined.iloc[idx_test,
                                                  :].iloc[idx_val, :],
                             test=Updated_test,
                             complete=Ag_combined)

    return Seq_Ag_data  # 这里返回的是训练集、验证集、测试集和完整数据集。

# 定义第十二个函数，seq_classification，这个用来模拟生成7.27×10的7次方的序列。
def seq_classification(classifier, flatten_input=False):
    """
    体外生成序列并将其分类为结合或非结合序列。
    
    Parameters
    ---
    classifier: 使用的神经网络分类模型。

    flatten_input: 如果为 True，体外生成的序列输入在分类前会被展平。这对于接受二维输入的神经网络（即 ANN）是必要的。

    Returns
    ---
    pos_seq, pos_pred: 包含所有正样本序列及其预测值的数组。
    """

    # 定义每个位置允许的氨基酸 3 16 16 15 3 3 1 9 6 13，相乘即为7.27×10的7次方
    AA_per_pos = [
        ['F', 'Y', 'W'],
        ['A', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V'],
        ['A', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V'],
        ['A', 'C', 'D', 'F', 'G', 'H', 'I', 'L',
         'N', 'P', 'R', 'S', 'T', 'V', 'Y'],
        ['A', 'G', 'S'],
        ['F', 'L', 'M'],
        ['Y'],
        ['A', 'E', 'K', 'L', 'M', 'P', 'Q', 'T', 'V'],
        ['F', 'H', 'I', 'L', 'N', 'Y'],
        ['A', 'D', 'E', 'H', 'I', 'K', 'L', 'M',
         'N', 'P', 'Q', 'T', 'V']]

    # 循环中使用的参数
    current_seq = np.empty(0, dtype=object)
    dim = [len(x) for x in AA_per_pos]
    idx = [0]*len(dim)
    counter = 1
    pos = 0

    # 存储结果的数组
    pos_seq = np.empty(0, dtype=str)
    pos_pred = np.empty(0, dtype=float)

    while(1):
        # 获取所有可能的氨基酸组合
        l_comb = []
        for i in range(0, len(dim)):
            l_comb.append(AA_per_pos[i][idx[i]])

        # 添加当前序列
        current_seq = np.append(current_seq, (''.join(l_comb)))

        # 每500个序列运行分类
        if len(current_seq) == 500:
            # 运行当前序列的分类
            seq_pred = run_classification(
                current_seq, classifier, flatten_input
            )

            # 加显著序列和预测值
            pos_seq = np.append(
                pos_seq, current_seq[np.where(seq_pred > 0.50)[0]]
            )
            pos_pred = np.append(
                pos_pred, seq_pred[np.where(seq_pred > 0.50)[0]]
            )

            # 清空当前序列数组
            current_seq = np.empty(0, dtype=object)

            # 打印进度条
            progbar(counter, np.ceil(np.prod(dim)/500))
            counter += 1

        # 终止条件
        if sum(idx) == (sum(dim)-len(dim)):
            # 运行当前序列的分类
            seq_pred = run_classification(
                current_seq, classifier, flatten_input
            )

            # 添加显著序列和预测值
            pos_seq = np.append(
                pos_seq, current_seq[np.where(seq_pred > 0.50)[0]]
            )
            pos_pred = np.append(
                pos_pred, seq_pred[np.where(seq_pred > 0.50)[0]]
            )

            break

        # 更新索引
        while(1):
            if (idx[pos]+1) == dim[pos]:
                idx[pos] = 0
                pos += 1
            else:
                idx[pos] += 1
                pos = 0
                break

    return pos_seq, pos_pred

# 创建包含训练集和测试集的集合
mHER_H3_all = data_split(mHER_H3_AgPos, mHER_H3_AgNeg)  # 使用上述函数，将数据集分为了训练集、验证集和测试集（70，15，15）。
                                                        # 并将测试集调整至90%负样本，10%正样本。

# 创建模型目录
model_dir = 'classification'
os.makedirs(model_dir, exist_ok=True)

# 使用在单独脚本中调整的CNN模型参数。
params = [['CONV', 400, 5, 1],
          ['DROP', 0.2],
          ['POOL', 2, 1],
          ['FLAT'],
          ['DENSE', 300]]

# 用未调整的（类别划分）数据集训练和测试CNN
CNN_all = CNN_classification(
    mHER_H3_all, 'All_data', save_model=model_dir, params=params
)

# 在计算机中生成 CDRH3 序列并计算它们的属性
# 如果 P(binder) > 0.5，则为预测值。
print('[INFO] Classifying in silico generated sequences')
CNN_all_seq, CNN_all_pred = seq_classification(CNN_all)
print('[INFO] Done')

# 将输出写入 .csv 文件
CNN_all_df = pd.DataFrame(
    {'AASeq': CNN_all_seq, 'Pred': CNN_all_pred}, columns=['AASeq', 'Pred']
)
CNN_all_df.to_csv(
    os.path.join('data/CNN_H3_all.csv')
)
