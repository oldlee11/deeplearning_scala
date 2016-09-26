package dp_process

import scala.util.Random
import scala.math

import java.io.File; 
import java.io._
import scala.io.Source


  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

//主要思路  使用dA做非监督学习精选数据降维,提取特征
//在使用监督学习训练输出层LR
class SdA(val n_ins: Int, hidden_layer_sizes: Array[Int], val n_outs: Int, val n_layers:Int, var rng: Random=null) {

  def sigmoid(x: Double): Double = {
    return 1.0 / (1.0 + math.pow(math.E, -x))
  }
  var input_size: Int = 0
  // var hidden_layer_sizes: Array[Int] = new Array[Int](n_layers)
  //n_layers个以sigmoid为映射函数的HiddenLayer
  var sigmoid_layers: Array[HiddenLayer] = new Array[HiddenLayer](n_layers)
  //n_layers个dA
  var dA_layers: Array[dA] = new Array[dA](n_layers)
  if(rng == null) rng = new Random(1234)
  var i: Int = 0
  // construct multi-layer
  //组合n层网络
  for(i <- 0 until n_layers) {
    if(i == 0) {
      input_size = n_ins
    } else {
      input_size = hidden_layer_sizes(i-1)
    }
    // construct sigmoid_layer
    sigmoid_layers(i) = new HiddenLayer(input_size, hidden_layer_sizes(i), null, null, rng)
    // construct dA_layer
    //注意dA_layers中每一层的系数W,bias和sigmoid_layers是共享的
    //注意,由于传入的是向量指针,所以在dA_layers(i) 训练更新w和bias的过程中,真实数据是跟着改变的(函数内部改变了共享的权值参数变量)
    dA_layers(i) = new dA(input_size, hidden_layer_sizes(i), sigmoid_layers(i).W, sigmoid_layers(i).b, null, rng)
  }
  // layer for output using LogisticRegression
  //输出层用于监督学习使用LR
  val log_layer = new LogisticRegression(hidden_layer_sizes(n_layers-1), n_outs)
  
  /*预训练,先对每一层dA逐层训练,由于sigmoid_layers和dA_layers的参数权值是共享的,所以训练完的A后就等于初始化了sigmoid_layers的参数
    lr:参数迭代的学习率
    corruption_level:加噪音的比例  =[0,1],值越高,加入噪音个数约多(原始数据输入由x->0)
   * */
  def pretrain(train_X: Array[Array[Double]], lr: Double, corruption_level: Double, epochs: Int,save_module_path:String="") {
    val N:Int=train_X.length
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input_size: Int = 0
    var prev_layer_input: Array[Double] = new Array[Double](0)
    var i: Int = 0
    var j: Int = 0
    var epoch: Int = 0
    var n: Int = 0
    var l: Int = 0
    //一层一层的训练dA(做非监督学习)
    for(i <- 0 until n_layers) {  // layer-wise
      println("对第"+i+"层的da进行预训练")//debug
      //每一层批量训练epochs次
      for(epoch <- 0 until epochs) {  // training epochs
        println("进行第"+epoch+"次批量训练")//debug
        dA_layers(i).cross_entropy_result=0//debug
        //每一层批量训练N次(N个训练样本)
        for(n <- 0 until N) {  // input x1...xN
          // layer input 计算每一层dA的输入数据
          for(l <- 0 to i) {
            if(l == 0) {
              //如果是训练第1层dA则layer_input就等于第n个样本输入的x
              layer_input = new Array[Double](n_ins)
              for(j <- 0 until n_ins) layer_input(j) = train_X(n)(j)
            } else {
              //设置上两层(上一层的上一层)的单元个数
              if(l == 1) {
                //如果是第2层
                prev_layer_input_size = n_ins
              }else {
                //如果是第3---层
                prev_layer_input_size = hidden_layer_sizes(l-2)
              }
              //把上次计算时的上一层=layer_input 赋值给上两层prev_layer_input
              prev_layer_input = new Array[Double](prev_layer_input_size)
              for(j <- 0 until prev_layer_input_size) prev_layer_input(j) = layer_input(j)
              //上一层数据layer_input重新定义为hidden_layer_sizes(l-1)个单元
              layer_input = new Array[Double](hidden_layer_sizes(l-1))
              //根据prev_layer_input经过 sigmoid_layers(l-1) 后输出为 layer_input
              sigmoid_layers(l-1).sample_h_given_v(prev_layer_input, layer_input)
            }
          }
          //使用layer_input 对第i层dA做训练
          dA_layers(i).train(layer_input, lr, corruption_level,N)
        }
        println("cross_entropy="+dA_layers(i).cross_entropy_result/N) //debug
      }
      //保存第i层da的参数
      if(! (save_module_path=="")){
        dA_layers(i).save_w(save_module_path+"SdA_mnist_module_da"+i+".txt") //debug 
      }
    }
  }

  //经过预训练完后(使用dA的参数 初始化了sigmoid_layers的参数),使用y值做监督学习
  def finetune(train_X: Array[Array[Double]], train_Y: Array[Array[Int]], lr: Double, epochs: Int,save_module_path:String="") {
    val N:Int=train_X.length
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input: Array[Double] = new Array[Double](0)
    var epoch: Int = 0
    var n: Int = 0
    //批量训练epochs次
    for(epoch <- 0 until epochs) {
      println("进行第"+epoch+"次批量训练")//debug
      log_layer.cross_entropy_result=0.0
      //每一层批量训练N次(N个训练样本)
      for(n <- 0 until N) {
        // layer input 依次经过每一层的sigmoid_layers 最后输出layer_input作为输出层log_layer的输入数据
        for(i <- 0 until n_layers) {
          if(i == 0) {
            prev_layer_input = new Array[Double](n_ins)
            for(j <- 0 until n_ins) prev_layer_input(j) = train_X(n)(j)
          } else {
            prev_layer_input = new Array[Double](hidden_layer_sizes(i-1))
            for(j <- 0 until hidden_layer_sizes(i-1)) prev_layer_input(j) = layer_input(j)
          }
          layer_input = new Array[Double](hidden_layer_sizes(i))
          sigmoid_layers(i).sample_h_given_v(prev_layer_input, layer_input)
        }
        //训练输出层log_layer
        log_layer.train(layer_input, train_Y(n), lr,N)
      }
      println(log_layer.cross_entropy_result/N)
      // lr *= 0.95
    }
    if(! (save_module_path=="")){
      log_layer.save_w(save_module_path+"SdA_mnist_module_LR.txt")
    }    
  }

  def predict(x: Array[Double]):Array[Double]= {
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input: Array[Double] = new Array[Double](n_ins)
    var j: Int = 0
    for(j <- 0 until n_ins) prev_layer_input(j) = x(j)
    var linear_output: Double = 0.0
    // layer activation
    var i: Int = 0
    var k: Int = 0
    for(i <- 0 until n_layers) {
      layer_input = new Array[Double](sigmoid_layers(i).n_out)
      for(k <- 0 until sigmoid_layers(i).n_out) {
        linear_output = 0.0
        for(j <- 0 until sigmoid_layers(i).n_in) {
          linear_output += sigmoid_layers(i).W(k)(j) * prev_layer_input(j)
        }
        linear_output += sigmoid_layers(i).b(k)
        layer_input(k) = sigmoid(linear_output)
      }
      if(i < n_layers-1) {
        prev_layer_input = new Array[Double](sigmoid_layers(i).n_out)
        for(j <- 0 until sigmoid_layers(i).n_out) prev_layer_input(j) = layer_input(j)
      }
    }
    val y:Array[Double]=new Array(log_layer.n_out)
    for(i <- 0 until log_layer.n_out) {
      y(i) = 0.0
      for(j <- 0 until log_layer.n_in) {
        y(i) += log_layer.W(i)(j) * layer_input(j)
      }
      y(i) += log_layer.b(i)
    }
    log_layer.softmax(y)
  }
}


object SdA {
  def test_SdA_simple() {
    val rng: Random = new Random(123)
    val pretrain_lr: Double = 0.1
    val corruption_level: Double = 0.3
    val pretraining_epochs: Int = 1000
    val finetune_lr: Double = 0.1
    val finetune_epochs: Int = 500
    val train_N: Int = 15
    val test_N: Int = 6
    val n_ins: Int = 36
    val n_outs: Int = 3
    val hidden_layer_sizes: Array[Int] = Array(15, 15)
    val n_layers: Int = hidden_layer_sizes.length
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
    // construct SdA
    val sda:SdA = new SdA(n_ins, hidden_layer_sizes, n_outs, n_layers, rng)
    // pretrain
    sda.pretrain(train_X, pretrain_lr, corruption_level, pretraining_epochs)
    // finetune
    sda.finetune(train_X, train_Y, finetune_lr, finetune_epochs)
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
    // test
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until test_N) {
      test_Y(i)=sda.predict(test_X(i))
      for(j <- 0 until n_outs) {
        print(test_Y(i)(j) + " ")
      }
      println()
    /*
0.9925479668371949 0.003462989872774893 0.003989043290030165 
0.9911440274737663 0.004638346251427853 0.004217626274805904 
0.00382429130659944 0.991158714101121 0.005016994592279654 
0.007577560047847417 0.9851308229664293 0.007291616985723298 
0.010036000223328788 0.004099123934047275 0.9858648758426238 
0.005976316112905411 0.007159145158669929 0.9868645387284246 
     * */      
    }
  }

  def train_SdA_mnist() {
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
    val pretrain_lr: Double = 0.1
    val corruption_level: Double = 0.1
    val pretraining_epochs: Int = 200
    val finetune_lr: Double = 0.1
    val finetune_epochs: Int = 300
    val n_ins: Int = 28*28
    val n_outs: Int = 10
    //val hidden_layer_sizes: Array[Int] = Array(300, 200, 100 ,50)
    val hidden_layer_sizes: Array[Int] = Array(500, 500, 2000)
    val n_layers: Int = hidden_layer_sizes.length
    val sda:SdA = new SdA(n_ins, hidden_layer_sizes, n_outs, n_layers, rng)
    
    //训练
    // pretrain
    sda.pretrain(train_X, pretrain_lr, corruption_level, pretraining_epochs,"D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//")
    // finetune
    sda.finetune(train_X, train_Y, finetune_lr, finetune_epochs,"D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//")    
  
  }    
  

  def test_SdA_mnist() {
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
        
    //建立sda   
    //val hidden_layer_sizes: Array[Int] = Array(300, 200, 100 ,50)//0.7873
    val hidden_layer_sizes: Array[Int] = Array(500, 500, 2000)//0.8817
    val n_layers: Int = hidden_layer_sizes.length    
    val rng: Random = new Random(123)
    val sda:SdA = new SdA(n_ins, hidden_layer_sizes, n_outs, n_layers, rng)
    //读取之前训练的模型
    for(i <-0 until n_layers){
      sda.dA_layers(i).read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//SdA_mnist_module_da"+i+".txt")  
    }
    sda.log_layer.read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//SdA_mnist_module_LR.txt")
    
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
    
    val writer_test_out = new PrintWriter(new File("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//test//SdA_mnist_test.txt"))
    var i: Int = 0
    var j: Int = 0
    var pred_right_nums:Int=0;
    for(i <- 0 until test_N) {
      test_Y_pred(i)=sda.predict(test_X(i))
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
    //test_SdA_simple()
    train_SdA_mnist()
    test_SdA_mnist()
  }

}
