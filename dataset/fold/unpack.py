import gzip
import six.moves.cPickle as pickle

# data loading functions
def atisfold():
    data_set_lex=list()
    data_set_ne=list()
    data_set_y=list()
    for fold in (0,1,2,3,4):
        filename = "D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//fold//"+"atis.fold"+str(fold)+".pkl.gz"
        f = gzip.open(filename, 'rb')
        try:
            train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set, dicts = pickle.load(f)
        #deal with dicts
        idx2label = dict((k, v) for v, k in dicts['labels2idx'].items())
        idx2word  = dict((k, v) for v, k in dicts['words2idx'].items())
        for data_set in (train_set, valid_set, test_set):
            data_set_lex_tmp, data_set_ne_tmp, data_set_y_tmp = data_set
            data_set_lex.extend([map(lambda x: idx2word[x], w) for w in data_set_lex_tmp])
            data_set_ne.extend(data_set_ne_tmp)
            data_set_y.extend([map(lambda x: idx2label[x], y) for y in data_set_y_tmp])
        f.close()
    return data_set_lex,data_set_ne,data_set_y

data_set_lex,data_set_ne,data_set_y=atisfold()


f=file("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//fold//fold.txt","w+")
for i in range(len(data_set_lex)):
    i_str="=sep1=".join(data_set_lex[i])+"=sep2="+"=sep1=".join(str(k) for k in data_set_ne[i])+"=sep2="+"=sep1=".join(data_set_y[i]);
    f.writelines(i_str+"\n");
f.close()



words=dict()
labels=dict()
k=0
m=0
for i in range(len(data_set_lex)):
    for j in range(len(data_set_lex[i])):
        if data_set_lex[i][j] not in words:
            words[data_set_lex[i][j]]=k   
            k=k+1 
        if data_set_y[i][j] not in labels:
            labels[data_set_y[i][j]]=m
            m=m+1


f=file("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//fold//fold_code.txt","w+")
for i in range(len(data_set_lex)):
    i_str="=sep1=".join([str(words[k]) for k in data_set_lex[i]])+"=sep2="+"=sep1=".join([str(k) for k in data_set_ne[i]])+"=sep2="+"=sep1=".join([str(labels[k]) for k in data_set_y[i]]);
    f.writelines(i_str+"\n");
f.close()


f=file("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//fold//fold_dict_words.txt","w+")
for key_i in words.keys():
    i_str=key_i+"=sep1="+str(words[key_i])
    f.writelines(i_str+"\n");
f.close()


f=file("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//fold//fold_dict_labels.txt","w+")
for key_i in labels.keys():
    i_str=key_i+"=sep1="+str(labels[key_i])
    f.writelines(i_str+"\n");
f.close()