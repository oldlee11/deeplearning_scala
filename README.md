# deeplearning_scala
deeplearning with scala(2.10.6)  java-SE 1.8

src/dp_process/* include :  
    LogisticRegression(逻辑回归层softmax输出)
    
    HiddenLayer(常规神经网络的隐藏层)
    
    Dropout(多层感知器MLP=多个HiddenLayer+LogisticRegression)
    
    dA(对Hiddenlayer的单层预训练dA方式)
    
    sdA(对MLP的多层预训练dA方式)
    
    RBM(对Hiddenlayer的单层预训练RBM方式)
    
    DBN(对MLP的多层预训练RBM方式)
    
    CNN(卷积神经网络)
    
    CNN+dA(卷积神经网络+dA方式预训练[不确定是否完全正确])
    
LogisticRegression(逻辑回归层softmax输出),HiddenLayer(常规神经网络的隐藏层),Dropout(多层感知器MLP=多个HiddenLayer+LogisticRegression),dA(对Hiddenlayer的单层预训练dA方式),sdA(对MLP的多层预训练dA方式),RBM(对Hiddenlayer的单层预训练RBM方式),DBN(对MLP的多层预训练RBM方式),CNN(卷积神经网络),CNN+dA(卷积神经网络+dA方式预训练[不确定是否完全正确])

do labs with minst dataset 

    sdA:88.17%,DBN:88.7%,CNN(lenet5):90.3%


src/dp_process_parallel/* include : Dropout(mlp) and CNN with parallel code


do labs with minst dataset 

    Dropout(mlp):86.22%,CNN(lenet5):96.08%
