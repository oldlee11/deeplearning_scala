package dp_process

import scala.util.Random
import scala.math

import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

class Dropout(val n_in:Int, hidden_layer_sizes:Array[Int], n_out:Int,var rng: Random=null, val activation:String="ReLU") {
  var input_size: Int = 0
  val n_layers:Int=hidden_layer_sizes.length
  // var hidden_layer_sizes: Array[Int] = new Array[Int](n_layers)
  //n_layers个HiddenLayer
  var hidden_layers: Array[HiddenLayer] = new Array[HiddenLayer](n_layers)
  if(rng == null) rng = new Random(1234)
  var i: Int = 0
  // construct multi-layer
  //组合n层网络
  for(i <- 0 until n_layers) {
    if(i == 0) {
      input_size = n_in
    } else {
      input_size = hidden_layer_sizes(i-1)
    }
    // construct hidden_layer  ,注意这里以ReLu作为非线性映射函数,而不是sigmoid
    hidden_layers(i) = new HiddenLayer(input_size, hidden_layer_sizes(i), null, null, rng,activation)
  }
  // layer for output using LogisticRegression
  //输出层用于监督学习使用LR
  val log_layer = new LogisticRegression(hidden_layer_sizes(n_layers-1), n_out)  
  
  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)
    x:一个样本的x数据,据取值为[0,1]之间
    y:样本的标准输出
    lr:参数迭代的学习率
    batch_num 是一次批量迭代使用的样本数,由于每一次训练时,样本仅输入1个,参数(W,vbias,hbias)迭代计算时：W(t+1)=W(t)+lr*(局部误差/batch_num)
                                                 即每一次迭代仅仅更新1/batch_num倍,如果batch_num=训练集样本个数,则当所有训练集都迭代一遍后则 为 W(t+1)=W(t)+lr*(局部误差)
                                                如果batch_num=1则表示每个样本迭代一次 W(t+1)=W(t)+lr*(局部误差) 即不是批量迭代而是随机迭代了  
  */
  def train(x: Array[Double],y:Array[Int], lr: Double,L2_reg:Double=0.0,alpha:Double=0.9,batch_num:Int,dropout:Boolean=true, p_dropout:Double=0.5,debug:Boolean=false) {  
    /*
     * step1 forward hidden_layers
     */
    var dropout_masks:ArrayBuffer[Array[Double]]=ArrayBuffer()
    var layer_inputs:ArrayBuffer[Array[Double]]=ArrayBuffer()
    var layer_input:Array[Double]=Array()
    for(i <-0 until n_layers){
      if(i==0) {
        /*for (j <-0 until x.length){
          layer_input(j)=x(j)  
        }*/  
        layer_input=x 
      }
      layer_inputs +=layer_input
      if(debug){
        print("step1: forward hidden_layers:\n")
        print("layer_"+i+"_layer_input:");layer_input.foreach { x => print(x+"\t")};print("\n")//debug 
        print("hidden_layers"+i+"_w(max):"+hidden_layers(i).W.reduce(_++_).reduce(math.max(_,_))+"\n");//debug
        print("hidden_layers"+i+"_w(min):"+hidden_layers(i).W.reduce(_++_).reduce(math.min(_,_))+"\n");//debug
        print("hidden_layers"+i+"_w(avg):"+hidden_layers(i).W.reduce(_++_).reduce(_+_)/hidden_layers(i).W.reduce(_++_).length.toDouble+"\n");//debug
      }
      layer_input=hidden_layers(i).forward(layer_input)//layer_input:本层的输入变为下一层的输出,注意layer_input会改变 ,但是layer_inputs内的元素是不变化的
      if(dropout){
         var mask:Array[Double] = hidden_layers(i).dropout(size=layer_input.length, p=1-p_dropout)
         for(i <-0 until layer_input.length){
           layer_input(i) *= mask(i)  
         }  
         dropout_masks += mask
      }
    } 
    
    //layer_input.foreach { x => print(x+"\t")};print("\n")//debug 
    /*
     * step2 forward & backward log_layer
     */
    if(debug){
      print("step2.1: forward log_layer:\n")
      print("log_layer_input:");layer_input.foreach { x => print(x+"\t")};print("\n")//debug 
      print("log_layer"+"_w(max):"+log_layer.W.reduce(_++_).reduce(math.max(_,_))+"\n");//debug
      print("log_layer"+"_w(min):"+log_layer.W.reduce(_++_).reduce(math.min(_,_))+"\n");//debug
      print("log_layer"+"_w(avg):"+log_layer.W.reduce(_++_).reduce(_+_)/log_layer.W.reduce(_++_).length.toDouble+"\n");//debug    
    }
    log_layer.train(x=layer_input,y=y,lr=lr,batch_num=batch_num,L2_reg=L2_reg,alpha=alpha)
    if(debug){
      print("step2.2: backward log_layer:\n")
      print("log_layer"+"_dy:");log_layer.d_y.foreach { x =>print(x+"\t")};print("\n")//debug 
      print("log_layer"+"_dy(max):"+log_layer.d_y.reduce(math.max(_,_))+"\n");//debug
      print("log_layer"+"_dy(min):"+log_layer.d_y.reduce(math.min(_,_))+"\n");//debug
      print("log_layer"+"_dy(avg):"+log_layer.d_y.reduce(_+_)/log_layer.d_y.length.toDouble+"\n");//debug
      print("log_layer"+"_w(max):"+log_layer.W.reduce(_++_).reduce(math.max(_,_))+"\n");//debug
      print("log_layer"+"_w(min):"+log_layer.W.reduce(_++_).reduce(math.min(_,_))+"\n");//debug
      print("log_layer"+"_w(avg):"+log_layer.W.reduce(_++_).reduce(_+_)/log_layer.W.reduce(_++_).length.toDouble+"\n");//debug       
    }   
    /*
     * step3 backward hidden_layers
     */   
    for(i <- (0 until n_layers).reverse){
      if (dropout){
        if (i == n_layers-1){
          //下一层hidden_layers(i+1)是LogisticRegression
          hidden_layers(i).backward_2(next_layer=log_layer,input=layer_inputs(i),batch_num=batch_num,lr=lr,L2_reg=L2_reg,alpha=alpha,dropout=true,mask=dropout_masks(i))
        } else{
          //下一层hidden_layers(i+1)是HiddenLayer
          hidden_layers(i).backward_1(next_layer=hidden_layers(i+1),input=layer_inputs(i),batch_num=batch_num,lr=lr,L2_reg=L2_reg,alpha=alpha,dropout=true,mask=dropout_masks(i))
        }         
      }else{
        if (i == n_layers-1){
          //下一层hidden_layers(i+1)是LogisticRegression
          hidden_layers(i).backward_2(next_layer=log_layer,input=layer_inputs(i),batch_num=batch_num,lr=lr,L2_reg=L2_reg,alpha=alpha)
        } else{
          //下一层hidden_layers(i+1)是HiddenLayer
          hidden_layers(i).backward_1(next_layer=hidden_layers(i+1),input=layer_inputs(i),batch_num=batch_num,lr=lr,L2_reg=L2_reg,alpha=alpha)       
        }         
      }
      if(debug){
        print("step3 backward hidden_layers:\n")
        print("hidden_layers"+i+"_dv:");hidden_layers(i).d_v.foreach { x =>print(x+"\t")};print("\n")//debug
        print("hidden_layers"+i+"_dv(max):"+hidden_layers(i).d_v.reduce(math.max(_,_))+"\n");//debug
        print("hidden_layers"+i+"_dv(min):"+hidden_layers(i).d_v.reduce(math.min(_,_))+"\n");//debug
        print("hidden_layers"+i+"_dv(avg):"+hidden_layers(i).d_v.reduce(_+_)/hidden_layers(i).d_v.length.toDouble+"\n");//debug
        print("hidden_layers"+i+"_w(max):"+hidden_layers(i).W.reduce(_++_).reduce(math.max(_,_))+"\n");//debug
        print("hidden_layers"+i+"_w(min):"+hidden_layers(i).W.reduce(_++_).reduce(math.min(_,_))+"\n");//debug
        print("hidden_layers"+i+"_w(avg):"+hidden_layers(i).W.reduce(_++_).reduce(_+_)/hidden_layers(i).W.reduce(_++_).length.toDouble+"\n");//debug        
      }
    }
  }  
  
  
  /*训练,使用训练集做一个批次的BP训练
    lr:参数迭代的学习率
    inputs_x 训练集输入x
    inputs_y 训练集输出y
   * */  
  def train_batch(inputs_x: Array[Array[Double]], inputs_y: Array[Array[Int]],lr: Double,L2_reg:Double=0.0,alpha:Double=0.9,dropout:Boolean=true, p_dropout:Double=0.5,batch_num_per:Double=1.0,save_module_path:String="",debug:Boolean=false)={
    //抽取样本个数
    val batch_num:Int=if(batch_num_per==1.0){
      inputs_x.length
    }else{
      math.round((inputs_x.length*batch_num_per).toFloat)//每次批量训练样本数
    }
    log_layer.cross_entropy_result=0.0//每次批量开始时,把上次的交叉信息嫡清零  
    
    //完成一次批量训练
    val rng_epooch:Random=new Random()//每次生成一个种子
    val rng_index:ArrayBuffer[Int]=ArrayBuffer();
    if(batch_num_per==1.0){
      for(i <- 0 to (batch_num-1)) rng_index += i//抽样样本的角标 
    }else{
      for(i <- 0 to (batch_num-1)) rng_index += math.round((rng_epooch.nextDouble()*(inputs_x.length-1)).toFloat)//抽样样本的角标        
    }
    //正式训练一批次的样本
    for(i <- rng_index) {
      //根据一个样本完成一次训练,迭代增量取 1/batch_num
      if(debug){
        print("\n")//debug
      }
      train(x=inputs_x(i),y=inputs_y(i), lr=lr,L2_reg=L2_reg,batch_num=batch_num,dropout=dropout,p_dropout=p_dropout,debug=debug)
    }
    //完成一批次训练后计算本批次的平均交叉嫡cross_entropy_result/batch_num (cross_entropy_result内部是累加的)
    println("cross_entropy="+log_layer.cross_entropy_result/batch_num)    
    
      //保存第i层da的参数
    if(! (save_module_path=="")){
      hidden_layers(i).save_w(save_module_path+"drop_module_hidden"+i+".txt") //debug 
    }    
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
}  
  
object Dropout {
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
    val classifier = new  Dropout(n_in=9,hidden_layer_sizes=Array(100,50), n_out=n_out,rng=null,activation="ReLU")//实验成果了   hiddenlayer.scala 初始化 val a: Double =1/ math.pow(n_out,0.25)    learning_rate *=0.9   lr=0.1
    //val classifier = new  Dropout(n_in=9,hidden_layer_sizes=Array(100,50), n_out=n_out,rng=null,activation="sigmoid")//----没有成果过还
    val n_epochs:Int=50
    val train_N:Int=train_Y.length
    var learning_rate:Double=0.1
    // train
    var epoch: Int = 0
    var i: Int = 0
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate,L2_reg=0.0,alpha=0.9, dropout=true, p_dropout=0.1, batch_num_per=1.0, save_module_path="",debug=true)
      learning_rate *=0.9
    }
    
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
    val n_epochs: Int = 100
    val n_ins: Int = 28*28
    val n_outs: Int = 10
    val hidden_layer_sizes: Array[Int] = Array(500, 500, 2000)

    val classifier = new  Dropout(n_in=n_ins,hidden_layer_sizes=hidden_layer_sizes, n_out=n_outs,rng=rng,activation="ReLU")
    
    //使用sda或dbn产生的w和b初始化hidden
    for(i <-0 until hidden_layer_sizes.length){
      //classifier.hidden_layers(i).read_w_module_from_da_rbm("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//SdA_mnist_module_da"+i+".txt")
      classifier.hidden_layers(i).read_w_module_from_da_rbm("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//DBN_module_RBM"+i+".txt")
    }

    // train
    var epoch: Int = 0
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate,L2_reg=0.0,alpha=0.9, dropout=true, p_dropout=0.1, batch_num_per=0.01, save_module_path="D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//",debug=false)
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
    println(pred_right_nums.toDouble/(test_N.toDouble))//使用sda(da)=91%(原来是87%)  dbn(rbm)=90%(原来是87%)   
    
  }
  def main(args: Array[String]) {
    //test_Dropout_simple()
    train_tset_mnist_use_dbn_sda_dbn()
  }  
}



