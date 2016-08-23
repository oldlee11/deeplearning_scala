#C:\Users\liming>python D:\youku_work\python\spark_python_scala\scala\workpace\deeplearning\dataset\mnist\unpack.py


import pickle
import gzip

def load_data():
    with gzip.open('D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/mnist.pkl.gz') as fp:
        training_data, valid_data, test_data = pickle.load(fp)
    return training_data, valid_data, test_data

(training_data, valid_data, test_data)=load_data()

#len(valid_data[0])
#10000
#len(training_data[0])
#50000
#len(test_data[0])
#10000

def trans_txt(data_in,fileout):
    f=file(fileout,"w+")
    x=data_in[0];
    y=data_in[1]; 
    for i in range(len(x)):
        x_i_str=",".join([str(x_i) for x_i in x[i]]);
        f.writelines("=sep=".join([x_i_str,str(y[i])])+"\n");
    f.close() ;

trans_txt(training_data,'D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/training_data.txt')  
trans_txt(valid_data,'D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/valid_data.txt')  
trans_txt(test_data,'D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt')     