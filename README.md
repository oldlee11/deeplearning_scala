# deeplearning_scala
deeplearning with scala(2.10.6)  java-SE 1.8

  * deeplearning with scala and spark
  * Copyright liming(oldlee11)
  * Email: oldlee11@163.com
  * qq:568677413
   
<一> src/dp_process/* include :(串行基本方式实现)  

    LogisticRegression(逻辑回归层softmax输出)
    
    HiddenLayer(常规神经网络的隐藏层)
    
    Dropout(多层感知器MLP=多个HiddenLayer+LogisticRegression)
    
    dA(对Hiddenlayer的单层预训练dA方式)
    
    sdA(对MLP的多层预训练dA方式)
    
    RBM(对Hiddenlayer的单层预训练RBM方式)
    
    DBN(对MLP的多层预训练RBM方式)
    
    ConvPoolLayer(卷积层+max池化层)
    
    CNN(卷积神经网络=多个ConvPoolLayer+Dropout)
    
    RNN（多层卷积升级网络=多层RNN_HiddenLayer+1层RNN_LogisticRegression,没有实现lstm,支持多层的mlp的递归）
    
    CNN+dA(卷积神经网络+dA方式预训练[不确定是否完全正确])
    
    
    


使用minst数据集做实验

    sdA:88.17%
    
    DBN:88.7%
    
    CNN(lenet5):90.3%


<二> src/dp_process_parallel/* include :（并行方式实现） 

    Dropout_parallel=多个HiddenLayer_parallel+LogisticRegression_parallel
    
    CNN_parallel=多个ConvPoolLayer_parallel+Dropout_parallel

使用minst数据集做实验

    Dropout(mlp):86.22%
    
    CNN(lenet5):96.08%
