package dp_process_parallel

import scala.util.Random
import scala.math
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

class Dropout_parallel(n_in_arg:Int,
                       hidden_layer_sizes_arg:Array[Int], 
                       n_out:Int,
                       var rng: Random=null, 
                       activation:String="ReLU") {
  val hidden_layer_sizes:Array[Int]=hidden_layer_sizes_arg
  val n_in:Int=n_in_arg
  val n_layers:Int=hidden_layer_sizes.length
  
  // var hidden_layer_sizes: Array[Int] = new Array[Int](n_layers)
  //n_layers个HiddenLayer  
  var hidden_layers: Array[HiddenLayer_parallel] = new Array[HiddenLayer_parallel](n_layers)
  if(rng == null) rng = new Random(1234)
  // construct multi-layer
  //组合n层网络
  for(i <- 0 until n_layers) {
    var input_size: Int = 0
    if(i == 0) {
      input_size = n_in
    } else {
      input_size = hidden_layer_sizes(i-1)
    }
    // construct hidden_layer  ,注意这里以ReLu作为非线性映射函数,而不是sigmoid
    hidden_layers(i) = new HiddenLayer_parallel(input_size, hidden_layer_sizes(i), null, null, rng,activation)
    //print("hidden_layers_"+i+":"+hidden_layers(i).n_in+"\t"+hidden_layers(i).n_out+"\n")//debug
  }
  // layer for output using LogisticRegression
  //输出层用于监督学习使用LR
  val log_layer = new LogisticRegression_parallel(hidden_layer_sizes(n_layers-1), n_out)  
  //print("log_layer:"+log_layer.n_in+"\t"+log_layer.n_out+"\n")//debug
  

  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)
    x:一个样本的x数据,据取值为[0,1]之间
    y:样本的标准输出
             输出(Array[每一层的W_add_tmp,包括hidden和logistic层],
       Array[每一层的b_add_tmp,包括hidden和logistic层],
       cross_entropy值,
                       第一层hidden网络的d_v(如果是mlp则没有用处,如果是cnn则用于maxpool的反向梯度))
  */
  def train(x: Array[Double],
            y:Array[Int],
            dropout:Boolean=true, 
            p_dropout:Double=0.5):(Array[Array[Array[Double]]],Array[Array[Double]],Double,Array[Double])={  
    val layers_train_W_add_tmp:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()
    val layers_train_b_add_tmp:ArrayBuffer[Array[Double]]=ArrayBuffer()  
    /*
     * step1 forward hidden_layers
     */
    var dropout_masks:ArrayBuffer[Array[Double]]=ArrayBuffer()
    var layer_inputs:ArrayBuffer[Array[Double]]=ArrayBuffer()
    var layer_input:Array[Double]=Array()
    for(i <-0 until n_layers){
      if(i==0) {
        layer_input=x 
      }
      layer_inputs +=layer_input
      //print("hidden_layer_"+i+"_forward\n")//debug
      layer_input=hidden_layers(i).forward(layer_input)//layer_input:本层的输入变为下一层的输出,注意layer_input会改变 ,但是layer_inputs内的元素是不变化的
      if(dropout){
         var mask:Array[Double] = hidden_layers(i).dropout(size=layer_input.length, p=1-p_dropout)
         for(i <-0 until layer_input.length){
           layer_input(i) *= mask(i)  
         }  
         dropout_masks += mask
      }
    } 
    /*
     * step2 forward & backward log_layer
     */
    //print("log_layer train\n")//debug
    val log_layer_train=log_layer.train(x=layer_input,y=y)
    layers_train_W_add_tmp +=log_layer_train._1
    layers_train_b_add_tmp +=log_layer_train._2
    val log_layer_train_d_y:Array[Double]=log_layer_train._3
    val train_cross_entropy_result:Double=log_layer_train._4
    /*
     * step3 backward hidden_layers
     */  
    var hidden_layers_i_train_d_v:Array[Double]=Array() 
    for(i <- (0 until n_layers).reverse){
      //print("hidden_layer_"+i+"_backward\n")//debug
      if (i == n_layers-1){
        //下一层hidden_layers(i+1)是LogisticRegression
        val hidden_layers_i_train=hidden_layers(i).backward(input=layer_inputs(i),
                                                             next_layer_n_in=hidden_layer_sizes(n_layers-1),
                                                             next_layer_n_out=n_out,
                                                             next_layer_w=log_layer.W,
                                                             next_layer_dv=log_layer_train_d_y,
                                                             dropout=dropout,
                                                             mask=(if(dropout) dropout_masks(i) else Array()));  
        layers_train_W_add_tmp +=hidden_layers_i_train._1
        layers_train_b_add_tmp +=hidden_layers_i_train._2
        hidden_layers_i_train_d_v=hidden_layers_i_train._3
      }else{
        //下一层hidden_layers(i+1)是HiddenLayer
        val hidden_layers_i_train=hidden_layers(i).backward(input=layer_inputs(i),
                                                              next_layer_n_in= hidden_layer_sizes(i),
                                                              next_layer_n_out=hidden_layer_sizes(i+1),
                                                              next_layer_w=hidden_layers(i+1).W,
                                                              next_layer_dv=hidden_layers_i_train_d_v,
                                                              dropout=dropout,
                                                              mask=(if(dropout) dropout_masks(i) else Array()))   
        layers_train_W_add_tmp +=hidden_layers_i_train._1
        layers_train_b_add_tmp +=hidden_layers_i_train._2
        hidden_layers_i_train_d_v=hidden_layers_i_train._3   
      }      
    }
    (layers_train_W_add_tmp.reverse.toArray,layers_train_b_add_tmp.reverse.toArray,train_cross_entropy_result,hidden_layers_i_train_d_v)
  }    
  
  
  /*训练,使用训练集做一个批次的BP训练
    lr:参数迭代的学习率
    inputs_x 训练集输入x
    inputs_y 训练集输出y
   * */  
  def train_batch(inputs_x: Array[Array[Double]], 
                  inputs_y: Array[Array[Int]],
                  lr: Double,
                  alpha:Double=0.9,
                  dropout:Boolean=true, 
                  p_dropout:Double=0.5,
                  batch_num_per:Double=1.0,
                  debug:Boolean=false)={
    /*
     * 抽样数据
     * */
    //抽取样本个数
    val batch_num:Int=if(batch_num_per==1.0){
      inputs_x.length
    }else{
      math.round((inputs_x.length*batch_num_per).toFloat)//每次批量训练样本数
    }
    val rng_epooch:Random=new Random()//每次生成一个种子
    val rng_index:ArrayBuffer[Int]=ArrayBuffer();
    if(batch_num_per==1.0){
      for(i <- 0 to (batch_num-1)) rng_index += i//抽样样本的角标 
    }else{
      for(i <- 0 to (batch_num-1)) rng_index += math.round((rng_epooch.nextDouble()*(inputs_x.length-1)).toFloat)//抽样样本的角标        
    }
    
    /*
     * mapper
     * 计算参数的add_tmp
     * */
    /*串行
     val train_batch_result=rng_index.map(i=>train(x=inputs_x(i),
                                                          y=inputs_y(i),
                                                          dropout=dropout, 
                                                          p_dropout=p_dropout))*/
    /*并行*/
    val train_batch_result=rng_index.par.map(i=>train(x=inputs_x(i),
                                                          y=inputs_y(i),
                                                          dropout=dropout, 
                                                          p_dropout=p_dropout))
                                                          
    /*
     * reduce
     * 更新参数
     * */
    //w
    val W_add_tmp_all:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()
    var w_rows:Int=0;
    var w_cols:Int=0;
    for(i_layer<-0 until n_layers+1){//第i_layer层网络
      w_rows=train_batch_result(0)._1(i_layer).length
      w_cols=train_batch_result(0)._1(i_layer)(0).length
      W_add_tmp_all+=Array.ofDim[Double](w_rows,w_cols)
      for(w_row<-0 until w_rows){//第i_layer层网络中W的行数目
        for(w_col<-0 until w_cols){//第i_layer层网络中W的列数目
          for(i_train_time<-0 until train_batch_result.length){//第i_train_time个样本的训练结果
            W_add_tmp_all(i_layer)(w_row)(w_col) += train_batch_result(i_train_time)._1(i_layer)(w_row)(w_col)
          }
          W_add_tmp_all(i_layer)(w_row)(w_col)=W_add_tmp_all(i_layer)(w_row)(w_col)/batch_num//变化的平均值
          if(i_layer== n_layers){//logicit
            log_layer.W_add(w_row)(w_col)=alpha*log_layer.W_add(w_row)(w_col)+lr*W_add_tmp_all(i_layer)(w_row)(w_col)
            log_layer.W(w_row)(w_col) +=  log_layer.W_add(w_row)(w_col)            
          }else{
            hidden_layers(i_layer).W_add(w_row)(w_col)=alpha*hidden_layers(i_layer).W_add(w_row)(w_col)+lr*W_add_tmp_all(i_layer)(w_row)(w_col)
            hidden_layers(i_layer).W(w_row)(w_col) +=  hidden_layers(i_layer).W_add(w_row)(w_col)
          }
        }
      }
    }
    //b
    val b_add_tmp_all:ArrayBuffer[Array[Double]]=ArrayBuffer()
    var b_rows:Int=0;    
    for(i_layer<-0 until n_layers+1){//第i_layer层网络
      b_rows=train_batch_result(0)._2(i_layer).length
      b_add_tmp_all+=new Array[Double](b_rows)
      for(b_row<-0 until b_rows){//第i_layer层网络中b的行数目
        for(i_train_time<-0 until train_batch_result.length){//第i_train_time个样本的训练结果
          b_add_tmp_all(i_layer)(b_row) += train_batch_result(i_train_time)._2(i_layer)(b_row)
        }
        b_add_tmp_all(i_layer)(b_row)=b_add_tmp_all(i_layer)(b_row)/batch_num//变化的平均值
        if(i_layer== n_layers){//logicit
            log_layer.b_add(b_row)=alpha*log_layer.b_add(b_row)+lr*b_add_tmp_all(i_layer)(b_row)
            log_layer.b(b_row) +=  log_layer.b_add(b_row)            
        }else{
            hidden_layers(i_layer).b_add(b_row)=alpha*hidden_layers(i_layer).b_add(b_row)+lr*b_add_tmp_all(i_layer)(b_row)
            hidden_layers(i_layer).b(b_row) +=  hidden_layers(i_layer).b_add(b_row)
        }        
      }
    }
    //完成一批次训练后计算本批次的平均交叉嫡cross_entropy_result/batch_num (cross_entropy_result内部是累加的)
    println("cross_entropy="+train_batch_result.map(x=>x._3).reduce(_+_)/batch_num)   
  }
  
  
  //预测一个样本
  //注意 没有返回值  ,直接改变y
  def predict(x: Array[Double], dropout:Boolean=false,p_dropout:Double=0.5):Array[Double]={
    var layer_input:Array[Double] = x
    for(i <- 0 until n_layers){
      if(dropout){
        for(m <-0 until hidden_layers(i).n_out){
          for(n <-0 until hidden_layers(i).n_in){
            hidden_layers(i).W(m)(n)=(1 - p_dropout)* hidden_layers(i).W(m)(n)   
          }
        }
      }
      layer_input=hidden_layers(i).forward(layer_input)
    }
    log_layer.predict(layer_input)
  }  

  
  /* for time debug
   * */
  var fun_begin_time:Long=0
  def deal_begin_time()={
    fun_begin_time=System.currentTimeMillis()
  }
  def deal_end_time(fun_name:String)={
    print(fun_name+" use "+(System.currentTimeMillis()-fun_begin_time)+"毫秒\n")
    fun_begin_time=0
  }    
  
}

object Dropout_parallel {
  def test_Dropout_simple() {
    val train_X: Array[Array[Double]] = Array(
      Array(1, 1, 1, 0, 0, 0,0,0,0),
      Array(1, 0, 1, 0, 0, 0,0,0,0),
      Array(1, 1, 1, 0, 0, 0,0,0,0),
      Array(0, 0, 1, 1, 1, 0,0,0,0),
      Array(0, 0, 1, 0, 1, 0,0,0,0),
      Array(0, 0, 1, 1, 1, 0,0,0,0),
      Array(0, 0, 0, 0, 0, 0,1, 1, 1),
      Array(0, 0, 1, 0, 0, 0,1, 1, 1),
      Array(1, 0, 0, 0, 0, 0,1, 0, 1)      
    )
    val train_Y: Array[Array[Int]] = Array(
      Array(1, 0,0),
      Array(1, 0,0),
      Array(1, 0,0),
      Array(0, 1,0),
      Array(0, 1,0),
      Array(0, 1,0),
      Array(0, 0,1),
      Array(0, 0,1),
      Array(0, 0,1)      
    )
    val n_out:Int=3
    val classifier = new  Dropout_parallel(n_in_arg=9,hidden_layer_sizes_arg=Array(100,50), n_out=n_out,rng=null,activation="ReLU")//实验成果了   hiddenlayer.scala 初始化 val a: Double =1/ math.pow(n_out,0.25)    learning_rate *=0.9   lr=0.1
    val n_epochs:Int=50
    val train_N:Int=train_Y.length
    var learning_rate:Double=0.1
    // train
    var epoch: Int = 0
    var i: Int = 0
    classifier.deal_begin_time()
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate,alpha=0.9, dropout=true, p_dropout=0.1, batch_num_per=1.0,debug=true)
      learning_rate *=0.9
    }
    classifier.deal_end_time("train_batch")
     // test data
    val test_X: Array[Array[Double]] = Array(
      Array(1, 1, 1, 0, 0, 0,0,0,0),
      Array(0, 0, 1, 1, 1, 0,0,0,0),
      Array(0, 0, 0, 0, 0, 0,1,1,1)
    )
    val test_N:Int=test_X.length
    val test_Y: Array[Array[Double]] = Array.ofDim[Double](test_N, n_out)  
    // test
    var j: Int = 0
    for(i <- 0 until test_N) {
      test_Y(i)=classifier.predict(test_X(i))
      for(j <- 0 until n_out) {
        printf("%.5f ", test_Y(i)(j))
      }
      println()
    }    
  }  

  def train_tset_mnist_use_dbn_sda_dbn() {
    //读取训练集数据
    //val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/training_data.txt"
    val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/valid_data.txt"//train数据量大,暂时使用valid数据
    val train_X:Array[Array[Double]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>x._2)
    //例如    0---->Array(1,0,0,0,0,0,0,0,0,0)
    //    9---->Array(0,0,0,0,0,0,0,0,0,1)
    def trans_int_to_bin(int_in:Int):Array[Int]={
      val result:Array[Int]=Array(0,0,0,0,0,0,0,0,0,0);
      result(int_in)=1
      result
    }
    val train_Y:Array[Array[Int]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>trans_int_to_bin(x._1))
    val train_N: Int = train_X.length
    
    //设置da的基础参数并建立da
    val rng: Random = new Random(123)
    var learning_rate: Double = 0.001//学习率不能=0.1  由于已经初始化 所以是微调 所以是很小的lr
    val n_epochs: Int = 200
    val n_ins: Int = 28*28
    val n_outs: Int = 10
    val hidden_layer_sizes: Array[Int] = Array(500, 500, 2000)

    val classifier = new  Dropout_parallel(n_in_arg=n_ins,hidden_layer_sizes_arg=hidden_layer_sizes, n_out=n_outs,rng=rng,activation="ReLU")
    
    //使用sda或dbn产生的w和b初始化hidden
    for(i <-0 until hidden_layer_sizes.length){
      //classifier.hidden_layers(i).read_w_module_from_da_rbm("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//SdA_mnist_module_da"+i+".txt")
      classifier.hidden_layers(i).read_w_module_from_da_rbm("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//DBN_module_RBM"+i+".txt")//200次    87。05%
    }

    // train
    var epoch: Int = 0
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate,alpha=0.9, dropout=true, p_dropout=0.1, batch_num_per=0.01)
      learning_rate = learning_rate*0.9
    }    
    
    /*
     * test
     * */
    val filePath_test:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt"  
    val test_X:Array[Array[Double]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>x._2)
    val test_N: Int = test_X.length
    val test_Y_pred: Array[Array[Double]] = Array.ofDim[Double](test_N, n_outs)
    val test_Y: Array[Array[Int]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>trans_int_to_bin(x._1))
    
    //预测并对比
    //Array(0.1191,0.7101,0.0012)-->Array(0,1,0)
    def trans_pred_to_bin(pred_in:Array[Double]):Array[Int]={
      var max_index:Int=0
      for(i <-1 until pred_in.length){
        if(pred_in(i)>pred_in(max_index)){
          max_index=i
        }
      }
      val result:Array[Int]=Array(0,0,0,0,0,0,0,0,0,0);
      result(max_index)=1
      result      
    }
    
    //Array(1,0,0,0,0,0,0,0,0,0)---->0
    def trans_bin_to_int(bin_in:Array[Int]):Int={
      var result:Int= -1;
      for(i <-0 until bin_in.length){
        if(bin_in(i)==1){
          result=i
        }
      }
      result
    }
    
    var pred_right_nums:Int=0;
    for(i <- 0 until test_N) {
      test_Y_pred(i)=classifier.predict(test_X(i))
      print("第"+i+"个样本实际值:\n")
      print(test_Y(i).mkString(sep=","))
      print("\n")        
      print("第"+i+"个样本预测值:\n")
      print(test_Y_pred(i).mkString(sep=","))
      print("\n") 
      if(trans_bin_to_int(trans_pred_to_bin(test_Y_pred(i)))==trans_bin_to_int(test_Y(i))){
        pred_right_nums +=1
      }
    }
    println(pred_right_nums.toDouble/(test_N.toDouble))
    
  }  
  
  def train_tset_mnist_dbn() {
    //读取训练集数据
    //val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/training_data.txt"
    val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/valid_data.txt"//train数据量大,暂时使用valid数据
    val train_X:Array[Array[Double]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>x._2)
    //例如    0---->Array(1,0,0,0,0,0,0,0,0,0)
    //    9---->Array(0,0,0,0,0,0,0,0,0,1)
    def trans_int_to_bin(int_in:Int):Array[Int]={
      val result:Array[Int]=Array(0,0,0,0,0,0,0,0,0,0);
      result(int_in)=1
      result
    }
    val train_Y:Array[Array[Int]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>trans_int_to_bin(x._1))
    val train_N: Int = train_X.length
    
    //设置da的基础参数并建立da
    val rng: Random = new Random(123)
    var learning_rate: Double = 0.1//学习率不能=0.1  由于已经初始化 所以是微调 所以是很小的lr
    val n_epochs: Int = 100
    val n_ins: Int = 28*28
    val n_outs: Int = 10
    val hidden_layer_sizes: Array[Int] = Array(500, 500, 2000)

    val classifier = new  Dropout_parallel(n_in_arg=n_ins,hidden_layer_sizes_arg=hidden_layer_sizes, n_out=n_outs,rng=rng,activation="ReLU")

    // train
    var epoch: Int = 0
    classifier.deal_begin_time()
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate,alpha=0.9, dropout=true, p_dropout=0.1, batch_num_per=0.01)//lr=0.1  lr不变 迭代次数100 86.22%
      //learning_rate = learning_rate*0.9
    }    
    classifier.deal_end_time("train_batch")
    /*
     * test
     * */
    val filePath_test:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt"  
    val test_X:Array[Array[Double]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>x._2)
    val test_N: Int = test_X.length
    val test_Y_pred: Array[Array[Double]] = Array.ofDim[Double](test_N, n_outs)
    val test_Y: Array[Array[Int]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>trans_int_to_bin(x._1))
    
    //预测并对比
    //Array(0.1191,0.7101,0.0012)-->Array(0,1,0)
    def trans_pred_to_bin(pred_in:Array[Double]):Array[Int]={
      var max_index:Int=0
      for(i <-1 until pred_in.length){
        if(pred_in(i)>pred_in(max_index)){
          max_index=i
        }
      }
      val result:Array[Int]=Array(0,0,0,0,0,0,0,0,0,0);
      result(max_index)=1
      result      
    }
    
    //Array(1,0,0,0,0,0,0,0,0,0)---->0
    def trans_bin_to_int(bin_in:Array[Int]):Int={
      var result:Int= -1;
      for(i <-0 until bin_in.length){
        if(bin_in(i)==1){
          result=i
        }
      }
      result
    }
    
    var pred_right_nums:Int=0;
    for(i <- 0 until test_N) {
      test_Y_pred(i)=classifier.predict(test_X(i))
      print("第"+i+"个样本实际值:\n")
      print(test_Y(i).mkString(sep=","))
      print("\n")        
      print("第"+i+"个样本预测值:\n")
      print(test_Y_pred(i).mkString(sep=","))
      print("\n") 
      if(trans_bin_to_int(trans_pred_to_bin(test_Y_pred(i)))==trans_bin_to_int(test_Y(i))){
        pred_right_nums +=1
      }
    }
    println(pred_right_nums.toDouble/(test_N.toDouble))
  }
  def main(args: Array[String]) {
    //test_Dropout_simple()//串行使用543毫秒;并行使用387毫秒
    //train_tset_mnist_use_dbn_sda_dbn()
    train_tset_mnist_dbn()//串行使用 毫秒;并行使用 毫秒
  }    
}  