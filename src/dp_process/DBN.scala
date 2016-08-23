package dp_process

import scala.util.Random
import scala.math

import java.io.File; 
import java.io._
import scala.io.Source

import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

/*
 * n_ins:样本自变量的变量数(输入单元的个数)
 * hidden_layer_sizes:每一层隐藏层的单元个数(主要是rbm和hiddenlayer,做非监督自学习)
 * n_outs:样本y变量数(一般使用0,1表示多个分类)
 * n_layers:做监督学习的logisticregression,使用隐藏层的输出学习样本y分类变量
 * */
class DBN(val n_ins: Int, hidden_layer_sizes: Array[Int], val n_outs: Int, val n_layers: Int, var rng: Random=null) {

  def sigmoid(x: Double): Double = {
    return 1.0 / (1.0 + math.pow(math.E, -x))
  }

  var input_size: Int = 0
  var sigmoid_layers: Array[HiddenLayer] = new Array[HiddenLayer](n_layers)
  var rbm_layers: Array[RBM] = new Array[RBM](n_layers)
  if(rng == null) rng = new Random(1234)
  
  // construct multi-layer 组合隐藏层结构
  var i: Int = 0
  //依次初始化每一层的隐藏层网络
  for(i <- 0 until n_layers) {
    //隐藏层的输入=如果是第0层,则等于输入x的大小=n_ins
    //         如果是第1-n层,则等于上一层隐藏层的单元个数(输出)=hidden_layer_sizes(i-1)
    if(i == 0) {
      input_size = n_ins
    } else {
      input_size = hidden_layer_sizes(i-1)
    }
    // construct sigmoid_layer 初始化HiddenLayer
    sigmoid_layers(i) = new HiddenLayer(input_size, hidden_layer_sizes(i), null, null, rng)
    // construct rbm_layer  初始化RBM
    // 注意,RBM的W和hbias(b)权重系数同HiddenLayer的权重系数共享,即在预训练RBM后,HiddenLayer的系数也随之变化
    rbm_layers(i) = new RBM(input_size, hidden_layer_sizes(i), sigmoid_layers(i).W, sigmoid_layers(i).b, null, rng)
  }

  // layer for output using LogisticRegression 
  //初始化输出层,输入个数=最后一个隐藏层的单元个数=hidden_layer_sizes(n_layers-1),输出个数=样本y个数=n_outs
  val log_layer: LogisticRegression = new LogisticRegression(hidden_layer_sizes(n_layers-1), n_outs)

  /*
  //预训练每一层的RBM-原始=pretrain+pretrain_one_layer
  def pretrain(train_X: Array[Array[Double]], lr: Double, k: Int, epochs: Int) {
    val N:Int=train_X.length
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input_size: Int = 0
    var prev_layer_input: Array[Double] = new Array[Double](0)
    var i: Int = 0
    var j: Int = 0
    var epoch: Int = 0
    var n: Int = 0
    var l: Int = 0

    for(i <- 0 until n_layers) {  // layer-wise
      for(epoch <- 0 until epochs) {  // training epochs
        for(n <- 0 until N) {  // input x1...xN
          // layer input
          for(l <- 0 to i) {
            if(l == 0) {
              layer_input = new Array[Double](n_ins)
              for(j <- 0 until n_ins) 
                layer_input(j) = train_X(n)(j)
            } else {
              if(l == 1) 
                prev_layer_input_size = n_ins
              else 
                prev_layer_input_size = hidden_layer_sizes(l-2)
              prev_layer_input = new Array[Double](prev_layer_input_size)
              for(j <- 0 until prev_layer_input_size) 
                prev_layer_input(j) = layer_input(j)
              layer_input = new Array[Double](hidden_layer_sizes(l-1))
              sigmoid_layers(l-1).sample_h_given_v(prev_layer_input, layer_input)
            }
          }
          rbm_layers(i).contrastive_divergence(layer_input, lr, k,N)
        }
      }
    }
  }*/
  
  //新版改造
  def pretrain(train_X: Array[Array[Double]], lr: Double, k: Int, epochs: Int,batch_num_per:Double=1.0,save_module_path:String="") {
    for(i <- 0 until n_layers) {  // layer-wise
      pretrain_one_layer(i,train_X,lr,k,epochs,batch_num_per,save_module_path)
    }
  }
  
  //预训练某一层的rbm,新版添加的,sda没有
  /* 
   * layer_index对第几层隐藏层进行与训练
   * train_X 输入样本的x数据
   * lr 学习率
   * k cd算法使用
   * epochs 迭代次数
   * batch_num_per  从样本集合inputs中随机抽取一些样本,抽取的个数为样本总数*batch_num_per
   * */
  def pretrain_one_layer(layer_index:Int, train_X: Array[Array[Double]], lr: Double, k: Int, epochs: Int,batch_num_per:Double,save_module_path:String="") {
    println("对第"+layer_index+"层rbm做预训练")
    if(layer_index==0){
      println("    开始训练")
      //批量训练 epochs次
      for(epoch <- 0 until epochs) {
        println("第"+epoch+"次迭代:")
        rbm_layers(layer_index).contrastive_divergence_batch(train_X, lr, k,batch_num_per)
      }  
    }else{
      println("    开始对输入数据做隐藏层转化处理")
      var train_X_trans:Array[Array[Double]]=Array.ofDim[Double](train_X.length, hidden_layer_sizes(layer_index-1))
      //如果本次训练的不是第0层,则把输入样本数据 经过之前的每个隐藏层获取输出,作为本层隐藏层训练的输入
      var layer_input: Array[Double] = new Array[Double](0)//每一个样本经过layer_index以前的每一层隐藏层的sample_h_given_v函数处理后的结果
      var prev_layer_input_size: Int = 0
      var prev_layer_input: Array[Double] = new Array[Double](0)  
      //遍历每一个输入数据
      for(n <- 0 until train_X.length){
        //遍历layer_index以前的每一层隐藏层,使输入样本依次经过每个隐藏层的处理后数据=layer_input
        for(l <- 0 to layer_index) {
          if(l == 0) {
            //如果l=0 则layer_input=原始输入
            layer_input = new Array[Double](n_ins)
            for(j <- 0 until n_ins) 
              layer_input(j) = train_X(n)(j)
          }else {
            //如果l>0 则layer_input依次经过layer_index以前每一层隐藏层的sigmoid_layers(l-1).sample_h_given_v函数处理
            if(l == 1) 
              prev_layer_input_size = n_ins
            else 
              prev_layer_input_size = hidden_layer_sizes(l-2)
            prev_layer_input = new Array[Double](prev_layer_input_size)
            for(j <- 0 until prev_layer_input_size) 
              prev_layer_input(j) = layer_input(j)
            layer_input = new Array[Double](hidden_layer_sizes(l-1))
            sigmoid_layers(l-1).sample_h_given_v(prev_layer_input, layer_input)
          }
        } 
        //处理好layer_input后灌入train_X_trans中
        for (i<-0 until hidden_layer_sizes(layer_index-1))
          train_X_trans(n)(i)=layer_input(i)
      }
      println("    开始训练")
      //批量训练 epochs次
      for(epoch <- 0 until epochs) {
        println("第"+epoch+"次迭代:")
        rbm_layers(layer_index).contrastive_divergence_batch(train_X_trans, lr, k,batch_num_per)
      }  
    }
    //保存第i层rbm的参数
    if(! (save_module_path=="")){
      rbm_layers(layer_index).save_w(save_module_path+"DBN_module_RBM"+layer_index+".txt") //debug 
    }    
  }

  def finetune(train_X: Array[Array[Double]], train_Y: Array[Array[Int]], lr: Double, epochs: Int,save_module_path:String="") {
    val N:Int=train_X.length
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input: Array[Double] = new Array[Double](0)

    var epoch: Int = 0
    var n: Int = 0
    var i: Int = 0
    var j: Int = 0

    for(epoch <- 0 until epochs) {
      println("进行第"+epoch+"次批量训练")//debug
      log_layer.cross_entropy_result=0.0      
      for(n <- 0 until N) {
        // layer input
        for(i <- 0 until n_layers) {
          if(i == 0) {
            prev_layer_input = new Array[Double](n_ins)
            for(j <- 0 until n_ins) 
              prev_layer_input(j) = train_X(n)(j)
          } else {
            prev_layer_input = new Array[Double](hidden_layer_sizes(i-1))
            for(j <- 0 until hidden_layer_sizes(i-1)) 
              prev_layer_input(j) = layer_input(j)
          }
          layer_input = new Array[Double](hidden_layer_sizes(i))
          sigmoid_layers(i).sample_h_given_v(prev_layer_input, layer_input)
        }
        log_layer.train(layer_input, train_Y(n), lr,N)
      }
      println(log_layer.cross_entropy_result/N)      
      // lr *= 0.95
    }
    if(! (save_module_path=="")){
      log_layer.save_w(save_module_path+"DBN_module_LR.txt")
    }       
  }

  def predict(x: Array[Double]):Array[Double]= {
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input: Array[Double] = new Array[Double](n_ins)

    var i: Int = 0
    var j: Int = 0
    var k: Int = 0

    for(j <- 0 until n_ins) 
      prev_layer_input(j) = x(j)
    
    var linear_outoput: Double = 0
    // layer activation
    for(i <- 0 until n_layers) {
      layer_input = new Array[Double](sigmoid_layers(i).n_out)
      for(k <- 0 until sigmoid_layers(i).n_out) {
        linear_outoput = 0.0
        for(j <- 0 until sigmoid_layers(i).n_in) {
          linear_outoput += sigmoid_layers(i).W(k)(j) * prev_layer_input(j)
        }
        linear_outoput += sigmoid_layers(i).b(k)
        layer_input(k) = sigmoid(linear_outoput)
      }
      if(i < n_layers-1) {
        prev_layer_input = new Array[Double](sigmoid_layers(i).n_out)
        for(j <- 0 until sigmoid_layers(i).n_out) prev_layer_input(j) = layer_input(j)
      }
    }
    val y:Array[Double]=new Array[Double](log_layer.n_out)
    for(i <- 0 until log_layer.n_out) {
      y(i) = 0
      for(j <- 0 until log_layer.n_in) {
        y(i) += log_layer.W(i)(j) * layer_input(j)
      }
      y(i) += log_layer.b(i)
    }
    log_layer.softmax(y)
  }
}


object DBN {
  def test_dbn() {
    val rng: Random = new Random(123)
    val pretrain_lr: Double = 0.1
    val pretraining_epochs: Int = 1000
    val k: Int = 1
    val finetune_lr: Double = 0.1
    val finetune_epochs: Int = 500
    val train_N: Int = 15
    val test_N: Int = 6
    val n_ins: Int = 36
    val n_outs: Int = 3
    val hidden_layer_sizes: Array[Int] = Array(15, 15)
    val n_layers = hidden_layer_sizes.length

    // training data
    val train_X: Array[Array[Double]] = Array(
			Array(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1)			
    )
    val train_Y: Array[Array[Int]] = Array(
			Array(1, 0,0),
			Array(1, 0,0),
			Array(1, 0,0),
			Array(1, 0,0),
			Array(1, 0,0),
			Array(0, 1,0),
			Array(0, 1,0),
			Array(0, 1,0),
			Array(0, 1,0),
			Array(0, 1,0),
			Array(0, 0,1),
			Array(0, 0,1),
			Array(0, 0,1),
			Array(0, 0,1),
			Array(0, 0,1)			
    )
    // construct DBN
    val dbn: DBN = new DBN(n_ins, hidden_layer_sizes, n_outs, n_layers, rng)
		// pretrain
		dbn.pretrain(train_X, pretrain_lr, k, pretraining_epochs,1.0,"D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//");
		// finetune
		dbn.finetune(train_X, train_Y, finetune_lr, finetune_epochs,"D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//");
		// test data
    val test_X: Array[Array[Double]] = Array(
			Array(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1),
			Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1)						
    )
    val test_Y: Array[Array[Double]] = Array.ofDim[Double](test_N, n_outs)
    var i: Int = 0
    var j: Int = 0
    // test
    for(i <- 0 until test_N) {
      test_Y(i)=dbn.predict(test_X(i))
      for(j <- 0 until n_outs) {
        print(test_Y(i)(j) + " ")
      }
      println()
/*
0.9908222371732072 0.003056808629521544 0.006120954197271183 
0.9907251052126504 0.0030859661183220764 0.0061889286690275215 
0.0023919147994314644 0.9900787742611621 0.007529310939406653 
0.0021837970836596572 0.9897448342223722 0.00807136869396802 
0.005675775717707708 0.006750693506312952 0.9875735307759793 
0.005483547233821012 0.007079307903093865 0.9874371448630852 
 * */      
    }
  }
  

  def train_DBN_mnist() {
    //读取训练集数据
    val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/training_data.txt"
    //val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/valid_data.txt"//train数据量大,暂时使用valid数据
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
    
    //设置dbn的基础参数并建立dbn
    val rng: Random = new Random(123)
    val k: Int = 1
    val pretrain_lr: Double = 0.1
    val pretraining_epochs: Int = 200
    val finetune_lr: Double = 0.1
    val finetune_epochs: Int = 200
    val n_ins: Int = 28*28
    val n_outs: Int = 10
    //val hidden_layer_sizes: Array[Int] = Array(300, 200, 100 ,50)
    val hidden_layer_sizes: Array[Int] = Array(500, 500, 2000)//0.887
    val n_layers: Int = hidden_layer_sizes.length
    
    // construct DBN
    val dbn:DBN = new DBN(n_ins, hidden_layer_sizes, n_outs, n_layers, rng)
		// pretrain
		dbn.pretrain(train_X, pretrain_lr, k, pretraining_epochs,0.1,"D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//");
		// finetune
		dbn.finetune(train_X, train_Y, finetune_lr, finetune_epochs,"D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//");    
  }    
  

  def test_DBN_mnist() {
    //读取测试集数据
    val n_ins: Int = 28*28
    val n_outs: Int = 10       
    val filePath_test:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt"  
    val test_X:Array[Array[Double]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>x._2)
    val test_N: Int = test_X.length
    val test_Y_pred: Array[Array[Double]] = Array.ofDim[Double](test_N, n_outs)
    def trans_int_to_bin(int_in:Int):Array[Int]={
      val result:Array[Int]=Array(0,0,0,0,0,0,0,0,0,0);
      result(int_in)=1
      result
    }
    val test_Y: Array[Array[Int]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>trans_int_to_bin(x._1))
        
    //建立dbn   
    //val hidden_layer_sizes: Array[Int] = Array(300, 200, 100 ,50)
    val hidden_layer_sizes: Array[Int] = Array(500, 500, 2000)
    val n_layers: Int = hidden_layer_sizes.length    
    val rng: Random = new Random(123)
    val dbn:DBN = new DBN(n_ins, hidden_layer_sizes, n_outs, n_layers, rng)
    //读取之前训练的模型
    for(i <-0 until n_layers){
      dbn.rbm_layers(i).read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//DBN_module_RBM"+i+".txt")  
    }
    dbn.log_layer.read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//DBN_module_LR.txt")
    
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
    
    val writer_test_out = new PrintWriter(new File("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//test//DBN_mnist_test.txt"))
    var i: Int = 0
    var j: Int = 0
    var pred_right_nums:Int=0;
    for(i <- 0 until test_N) {
      test_Y_pred(i)=dbn.predict(test_X(i))
      writer_test_out.write("第"+i+"个样本实际值:\n")
      writer_test_out.write(test_Y(i).mkString(sep=","))
      writer_test_out.write("\n")        
      writer_test_out.write("第"+i+"个样本预测值:\n")
      writer_test_out.write(test_Y_pred(i).mkString(sep=","))
      writer_test_out.write("\n") 
      if(trans_bin_to_int(trans_pred_to_bin(test_Y_pred(i)))==trans_bin_to_int(test_Y(i))){
        pred_right_nums +=1
      }
    }
    println(pred_right_nums.toDouble/(test_N.toDouble))
    writer_test_out.close()
  }    

  def main(args: Array[String]) {
    //test_dbn()
    //train_DBN_mnist()
    test_DBN_mnist()
  }
}
