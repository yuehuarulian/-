<span style="font-family:宋体; font-size:10.5pt;">
<center><h1>人工智能课程设计Project 5: Machine Learning</h1></center>

<center><big></big></center>

## 一、问题概述

### 1.1 问题直观描述

- 这个项目需要实现感知器（Perceptron）和神经网络模型，进行手写数字识别和语言识别。
  - 手写数字识别用的多层感知机（Multilayer Perceptron, MLP）
  - 语言识别用的循环神经网络（RNN）

### 1.2 对项目已有代码的阅读和理解

- **autograder.py**

  - 实现作业评估

- **backend.py**

  - 实现数据集的划分
  - iterate_once：获取一个 batch_size 的数据
  - iterate_forever：循环不断地获取 batch_size 的数据，只有手动停止才能暂停
  - get_validation_accuracy：获取 val 数据集，只有手写数字识别和语言识别实现了

- **models.py**

  - 感知机、回归模型、数字识别、语言识别的具体实现
  - 包括 run get_loss train

- **nn.py**
  - 模型层次和操作的具体实现
  - Node DataNode FunctionNode：
    - Node 计算图中节点的基类，其包含节点类型、形状和节点在内存中的地址等信息。
    - DataNode 类是数据节点的基类，表示包含固定数据的节点，包括参数节点和常量节点。它包含一个数据属性和一个父节点列表。
    - FunctionNode 类是一个继承自 Node 的类，表示计算图中的函数节点，它根据其他节点计算其值，并执行必要的梯度计算和反向传播。在实例化时，需要提供其他节点作为输入，并检查输入节点的类型是否正确。该类还包含 data 属性，该属性通过调用 \_forward() 方法计算得到，以表示该节点的输出值。
  - Parameter：初始化矩阵
  - Constant：Constant 类是一个继承自 DataNode 的类，表示计算图中的常量节点，可以表示输入特征、输出标签或反向传播时的梯度值等常量数据。
  - Add AddBias DotProduct：矩阵操作，加法和点积
  - Linear：线性层，即感知机
  - ReLU：池化层
  - SquareLoss SoftmaxLoss：R2 损失函数和 softmax 损失函数
  - gradients(loss, parameters)：函数用于计算损失函数对于给定参数的梯度。
  - as_scalar：函数是用于将节点转换为标准 Python 数字的值

### 1.3 解决问题的思路和想法

## 二、算法设计与实现

### 2.1 感知机

#### 2.1.1 算法功能

- 感知器是一种二元分类器，用于将数据点分为特定类别（+1）或其他类别（-1）。

#### 2.1.2 设计思路

- 初始化：初始化感知器的权重向量。权重向量的维度与数据的维度相匹配。
- 得分计算：计算权重向量 self.w 与输入向量 x 的点积来计算感知器对数据点 x 的得分。
- 预测类别：根据得分来预测数据点 x 的类别。将得分转换为标量值，如果得分大于等于 0，则预测类别为 1，否则为-1。
- 训练过程：使用迭代器从给定的数据集中逐个获取数据点和对应的标签，进行预测并与真实标签进行比较。如果预测错误，则更新权重 self.w 以调整模型。
- 收敛判断：在训练过程中，通过检查是否存在预测错误的数据点来判断模型是否收敛。如果存在预测错误的数据点，则将 done 标记设置为 False，并继续训练直到所有数据点都被正确分类（即没有预测错误的数据点），此时 done 标记为 True，训练过程结束。

#### 2.1.3 代码实现

```py
class PerceptronModel(object):
    def __init__(self, dimensions):
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        return self.w

    def run(self, x):
        return nn.DotProduct(x, self.get_weights()) # features 1*c weight 1*c

    def get_prediction(self, x):
        score = nn.as_scalar(self.run(x))
        return 1 if score>=0 else -1

    def train(self, dataset:Dataset):
        batch_size = 1
        while True:
            finish = True
            for X, y in dataset.iterate_once(batch_size):
                y_pre = self.get_prediction(X)
                if y_pre != nn.as_scalar(y):
                    finish = False
                    self.w.update(X, nn.as_scalar(y))
            if finish:
                break

```

### 2.2 非线性回归(Non-linear Regression)

#### 2.2.1 算法功能

- 用于逼近从实数到实数的函数映射。使用神经网络来近似在区间[-2pi, 2pi]上的 sin(x)函数。

#### 2.2.2 设计思路

- 初始化：创建 nn.Parameter 对象来初始化模型的参数。这里使用 3 层全连接层，第一层有 50 个神经元，第二层有 20 个神经元，第三层输出 1
- 前向传播：将输入 x 通过神经网络进行计算，并返回预测的输出 y 。具体地，通过使用线性变换
  nn.Linear 和激活函数 nn.ReLU 对输入 x 进行处理，然后通过线性变换和偏置 nn.AddBias 得到最终
  的输出。
- 计算损失： get_loss 方法用于计算给定输入 x 和真实值 y 的损失值。这里使用平方损失函数（ nn.SquareLoss ），将模型的预测值 self.run(x) 和真实值 y 作为输入。
- 模型训练： train 方法用于训练模型。首先设置学习率（ learning_rate ）和批次大小（batch_size），然后通过一个无限循环迭代训练数据集。在每个迭代步骤中，从数据集中获取一个批次的输入和真实值，计算损失并获取梯度。然后使用梯度下降法更新模型的参数（权重和偏置），以减小损失值。循环终止的条件是整个数据集上的损失值低于 0.02。一旦满足条件，训练过程结束。

#### 2.2.3 代码实现

```py
class RegressionModel(object):
    def __init__(self):
        # Initialize your model parameters here
        self.learning_rate = 0.01
        self.batch_size = 10
        self.h1 = 50
        self.h2 = 20
        self.W1 = nn.Parameter(1, self.h1) # 1*50
        self.b1 = nn.Parameter(1, self.h1) # 1*50
        self.W2 = nn.Parameter(self.h1, self.h2) # 50*20
        self.b2 = nn.Parameter(1, self.h2) # 1*20
        self.W3 = nn.Parameter(self.h2, 1) # 20*1
        self.b3 = nn.Parameter(1, 1)     # 1*1
        # function: f(x) = ReLU(x * W1 + b1) * W2 + b2     x:batch_size*1

    def run(self, x):
        layer1 = nn.AddBias(nn.Linear(x, self.W1), self.b1) # b*1 1*50 1*50 -> b*50
        relu1 = nn.ReLU(layer1)

        layer2 = nn.AddBias(nn.Linear(relu1, self.W2), self.b2) # b*50 50*20 1*20 -> b*20
        relu2 = nn.ReLU(layer2)

        prediction = nn.AddBias(nn.Linear(relu2, self.W3), self.b3) # b*20 20*1 1*1 -> b*1
        return prediction

    def get_loss(self, x, y):
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        while True:
            for X, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(X,y)
                g_W1, g_b1, g_W2, g_b2, g_W3, g_b3 = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                self.W1.update(g_W1, -self.learning_rate)
                self.b1.update(g_b1, -self.learning_rate)
                self.W2.update(g_W2, -self.learning_rate)
                self.b2.update(g_b2, -self.learning_rate)
                self.W3.update(g_W3, -self.learning_rate)
                self.b3.update(g_b3, -self.learning_rate)
            if loss.data < 0.01:
                print(loss.data)
                break
```

### 2.3 手写数字识别

#### 2.3.1 算法功能

- 训练一个多层感知机来对 MNIST 数据集中的手写数字进行分类。

#### 2.3.2 设计思路

- 初始化：创建 nn.Parameter 对象来初始化模型的参数。

- 前向传播：输入 x 通过神经网络进行计算，并返回预测的输出。具体地，通过使用线性变换（nn.Linear）和激活函数（nn.ReLU）对输入 x 进行处理，然后通过线性变换和偏置（nn.AddBias）得到最终的输出。

- 计算损失：计算给定输入 x 和真实标签 y 的损失值。这里使用 softmax 损失函数，将模型的预测值 self.run(x) 和真实标签 y 作为输入。

- 模型训练： train 方法用于训练模型。首先设置学习率（ learning_rate ）和批次大小（batch_size），同时设置初始 epoch 为 0。然后通过一个无限循环迭代训练数据集。在每个迭代步骤中，从数据集中获取一个批次的输入和真实标签，计算损失并获取梯度。然后使用梯度下降法更新模型的参数（权重和偏置），以减小损失值。循环终止的条件是在验证集上达到了至少 98%的准确率。一旦满足条件，训练过程结束。

#### 2.3.3 代码实现

```py
class DigitClassificationModel(object):
    def __init__(self):
        # Initialize your model parameters here
        self.learning_rate = 0.02
        self.batch_size = 100
        self.h1 = 128
        self.h2 = 64
        self.figure = 10
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(28*28, self.h1) # 784*128
        self.b1 = nn.Parameter(1, self.h1) # 1*128
        self.W2 = nn.Parameter(self.h1, self.h2) # 128*32
        self.b2 = nn.Parameter(1, self.h2) # 1*32
        self.W3 = nn.Parameter(self.h2, self.figure) # 32*10
        self.b3 = nn.Parameter(1, self.figure)     # 1*10
        # function: f(x) = ReLU(x * W1 + b1) * W2 + b2     x:batch_size*1

    def run(self, x):
        layer1 = nn.AddBias(nn.Linear(x, self.W1), self.b1) # b*1 1*50 1*50 -> b*50
        relu1 = nn.ReLU(layer1)

        layer2 = nn.AddBias(nn.Linear(relu1, self.W2), self.b2) # b*50 50*20 1*20 -> b*20
        relu2 = nn.ReLU(layer2)

        prediction = nn.AddBias(nn.Linear(relu2, self.W3), self.b3) # b*20 20*1 1*1 -> b*1
        return prediction

    def get_loss(self, x, y):
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        while True:
            for X, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(X,y)
                g_W1, g_b1, g_W2, g_b2, g_W3, g_b3 = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                self.W1.update(g_W1, -self.learning_rate)
                self.b1.update(g_b1, -self.learning_rate)
                self.W2.update(g_W2, -self.learning_rate)
                self.b2.update(g_b2, -self.learning_rate)
                self.W3.update(g_W3, -self.learning_rate)
                self.b3.update(g_b3, -self.learning_rate)
                if dataset.get_validation_accuracy() >= 0.98:
                    return
```

### 2.4 语言识别

#### 2.4.1 算法功能

- 给定一段文本，确定该文本所使用的语言。用的 RNN，我还第一次尝试 RNN 的网络学到了
  ![alt text](image.png)

#### 2.4.2 设计思路

- 初始化：创建 nn.Parameter 对象来初始化模型的参数。模型的输入是一个包含 47 个唯一字符的字母表，同时有 5 种不同的语言需要识别。模型的隐藏层大小为 400。

- 前向传播：输入 xs 通过循环神经网络（RNN）进行计算，并返回预测的输出。在循环过程中，将每个字符的输入 x 与当前的隐藏状态 h 进行线性变换和激活函数处理，更新隐藏状态 h 。最终，通过线性变换和偏置计算输出得分。

- 计算损失：计算给定输入 xs 和真实标签 y 的损失值。这里使用 softmax 损失函数，将模型的预测值 self.run(xs) 和真实标签 y 作为输入。

- 模型训练：通过一个无限循环迭代训练数据集。在每个迭代步骤中，从数据集中获取一个批次的输入和真实标签，计算损失并获取梯度。然后使用梯度下降法更新模型的参数（权重和偏置），以减小损失值。循环终止的条件是在验证集上达到了至少 88%的准确率。一旦满足条件，训练过程结束。

#### 2.4.3 代码实现

```py
class LanguageIDModel(object):
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        """
        i = input size = 784 (because every input has 784 pixels)
        h = hidden layer size (variable, should test different values between 10 and 400)
        o = output size = 10 (because we classify for numbers 0 through 9)
        b = batch size (Hyperparameter, should evenly divide dataset (here 200))
        """
        self.figure = 5
        self.h = 200
        self.batch_size = 200
        self.learning_rate = 0.1
        """
        function: h_1(x) = ReLU([x * W])                      (batch_size,47)*(47*h) -> (batch_size,h)
        function: h_i(x) = ReLu([x * W] + [h_i * W_hidden])   (batch_size,47)*(47*h)+(batch_size,h)*(h,h) -> (batch_size,h)
        x  has dimensions   b x i
        W  has dimensions   i x h
        ==> h_1 = [x * W] has dimensions: b x h
        """
        self.W = nn.Parameter(self.num_chars, self.h)
        self.W_hidden = nn.Parameter(self.h, self.h)
        self.W_output = nn.Parameter(self.h, self.figure)

    def run(self, xs):
        L = len(xs)
        for i in range(L):
            if i == 0:
                h = nn.ReLU(nn.Linear(xs[i], self.W))
            else:
                h = nn.ReLU(nn.Add(nn.Linear(xs[i], self.W), nn.Linear(h, self.W_hidden)))
        return nn.Linear(h, self.W_output)

    def get_loss(self, xs, y):
        return nn.SquareLoss(self.run(xs), y)

    def train(self, dataset):
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                g_W, g_W_hidden, g_W_output = nn.gradients(loss, [self.W, self.W_hidden, self.W_output])
                self.W.update(g_W, -self.learning_rate)
                self.W_hidden.update(g_W_hidden, -self.learning_rate)
                self.W_output.update(g_W_output, -self.learning_rate)
            if dataset.get_validation_accuracy() > 0.84:
                break
```

## 三、实验结果

### 3.1 感知机

#### 3.1.1 测试截图

<img src="3d9af66d9501eca0bbeb4f4c1096b03e.png" style="zoom:50%;" />
<img src="183c4cfc9ccd30f3d05484c68fcf971e.png" style="zoom:50%;" />

### 3.2 非线性回归

#### 3.2.1 测试截图

<img src="85786185cef15c29dc65137ca440ed0e.png" style="zoom:50%;" />

- 修改了中间层，增大$h_1 h_2$，同时减小了退出的 loss 条件，改为 0.0001，可以看到拟合效果非常好

<img src="image-3.png" style="zoom:50%;" />

### 3.3 手写数字识别

#### 3.3.1 测试截图

<img src="image-1.png" style="zoom:50%;" />
<img src="image-2.png" style="zoom:50%;" />

### 3.4 语言识别

<img src="8776a28c7e48721ec2b93a6d88fe260c.png" alt="" style="zoom:50%;" />
<img src="2455ec5619750bfb421f20096d2ddae4.png" alt="" style="zoom:50%;" />

## 四、总结与分析

- 感知机是最简单的线性分类器，而且是二分类，同样多层感知机也用于处理线性分类问题。
  - 它的原理和 SVM 有点相似，都是通过$y = WX+b$来计算结果，train 的过程都是修正 W 的过程。但是 SVM 可以处理非线性，通过核函数把非线性的空间映射为线性空间，并且它的损失函数有正则化项和一个误分类惩罚项。而感知机只能线性分类，它通过结果来调整 W，它的损失函数是指导作用并没有明确的减少 loss 的机制。
- RNN 我之前没有了解过，不过它应该是语言方面最常用的神经网络之一，可能在于它有一个传递项，毕竟语言是由单词组成的，它需要建立每个单词之间的联系。它有一类特殊网络长短期记忆网络（LSTM），LSTM 通过引入三个门：输入门、遗忘门、输出门来控制信息的流动，这就比 RNN 只有一个$h_i$要传递更多的信息。
  </span>

<style>
    p {
        line-height: 18pt;
    }
</style>
