package dp_process

import scala.util.Random
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array
import scala.math

import java.io.File; 
import java.io._
import scala.io.Source

class CNN_dA(input_size:(Int,Int),
          output_size:Int,
          n_kernel_Array:Array[Int],
          kernel_size_Array:Array[(Int,Int)],
          pool_size_Array:Array[(Int,Int)],
          n_hidden:Int,
          n_channel:Int=3,
          _rng: Random=null,
          activation:String="ReLU",
          activation_mlp:String="tanh") extends CNN(input_size,
          output_size,
          n_kernel_Array,
          kernel_size_Array,
          pool_size_Array,
          n_hidden,
          n_channel,
          _rng,
          activation,
          activation_mlp){
  
  //建立ConvPoolLayer_layers
  val ConvPooldALayer_layers:Array[ConvPool_dA_Layer]= new Array[ConvPool_dA_Layer](n_ConvPoolLayer)
  for(i <-0 until n_ConvPoolLayer){
    if(i==0){  
      ConvPooldALayer_layers(i)=new ConvPool_dA_Layer(input_size_in=input_size,
                                                n_kernel_in=n_kernel_Array(i),
                                                kernel_size_in=kernel_size_Array(i),
                                                pool_size_in=pool_size_Array(i),
                                                n_channel_in=n_channel,
                                                _rng=rng,
                                                activation=activation)  
    }else{
      ConvPooldALayer_layers(i)=new ConvPool_dA_Layer(input_size_in=(ConvPoolLayer_layers(i-1).Max_PoolLayer_obj.s0,ConvPoolLayer_layers(i-1).Max_PoolLayer_obj.s1),
                                                n_kernel_in=n_kernel_Array(i),
                                                kernel_size_in=kernel_size_Array(i),
                                                pool_size_in=pool_size_Array(i),
                                                n_channel_in=ConvPoolLayer_layers(i-1).Max_PoolLayer_obj.pre_conv_layer_n_kernel,
                                                _rng=rng,
                                                activation=activation)        
    }
  }  
  
   /*
   * 用于 pretrain 预处理(SdA方法)
   * 模拟 SdA
   * */
  def binomial(n: Int, p: Double,rng:Random): Int = {
    if(p < 0 || p > 1) return 0
    var c: Int = 0
    var r: Double = 0.0
    var i: Int = 0
    for(i <- 0 until n) {
      r = rng.nextDouble()
      if(r < p) c += 1
    }
    return c
  }  
    /*预训练,先对每一层dA逐层训练,由于sigmoid_layers和dA_layers的参数权值是共享的,所以训练完的A后就等于初始化了sigmoid_layers的参数
    lr:参数迭代的学习率
    corruption_level:加噪音的比例  =[0,1],值越高,加入噪音个数约多(原始数据输入由x->0)
   * */
  def pretrain_sda_all(train_X: Array[Array[Array[Array[Double]]]], lr: Double, corruption_level: Double, epochs: Int,batch_num_per:Double=1.0,save_module_path:String="") {
    var lr_chang:Double=lr
    var layer_input: Array[Array[Array[Double]]] =Array() 
    var prev_layer_input_size: (Int,Int,Int) = (0,0,0)
    var prev_layer_input: Array[Array[Array[Double]]] = Array()
    val mlp_input_size_tmp:Int=ConvPoolLayer_layers(n_ConvPoolLayer-1).Max_PoolLayer_obj.s0 * ConvPoolLayer_layers(n_ConvPoolLayer-1).Max_PoolLayer_obj.s1  * ConvPoolLayer_layers(n_ConvPoolLayer-1).Max_PoolLayer_obj.pre_conv_layer_n_kernel
    val da_obj:dA=new dA(n_visible=mlp_input_size_tmp,n_hidden=n_hidden)//用于预训练输出层
    //抽取样本个数
    val batch_num:Int=if(batch_num_per==1.0){
      train_X.length
    }else{
      math.round((train_X.length*batch_num_per).toFloat)//每次批量训练样本数
    }
    //完成一次批量训练
    val rng_epooch:Random=new Random()//每次生成一个种子
    val rng_index:ArrayBuffer[Int]=ArrayBuffer();
    if(batch_num_per==1.0){
      for(i <- 0 to (batch_num-1)) rng_index += i//抽样样本的角标 
    }else{
      for(i <- 0 to (batch_num-1)) rng_index += math.round((rng_epooch.nextDouble()*(train_X.length-1)).toFloat)//抽样样本的角标        
    }   
    for(i <- 0 until (n_ConvPoolLayer+1)) {  
      println("对第"+i+"层进行预训练")//debug
      //每一层批量训练epochs次
      for(epoch <- 0 until epochs) {  // training epochs
        println("进行第"+epoch+"次批量训练")//debug
        if(i<n_ConvPoolLayer){
          ConvPooldALayer_layers(i).ConvLayer_obj.cross_entropy_result=0//debug
        }else{
          da_obj.cross_entropy_result=0
        }
        //每一层批量训练N次(N个训练样本)
        for(n <- rng_index) {  // input x1...xN
          // layer input 计算每一层的输入数据
          for(l <- 0 to i) {
            if(l == 0) {
              //如果是训练第1层则layer_input就等于第n个样本输入的x
              layer_input = Array.ofDim[Double](train_X(0).length,train_X(0)(0).length,train_X(0)(0)(0).length)
              for(i1<-0 until train_X(0).length){
                for(i2<-0 until train_X(0)(0).length){ 
                  for(i3<-0 until train_X(0)(0)(0).length){
                    layer_input(i1)(i2)(i3)=train_X(n)(i1)(i2)(i3)
                  }
                }
              }
            } else {
              //设置上两层(上一层的上一层)的单元个数
              if(l == 1) {
                //如果是第2层
                prev_layer_input_size = (train_X(0).length,train_X(0)(0).length,train_X(0)(0)(0).length)
              }else {
                //如果是第3---层
                prev_layer_input_size = (ConvPoolLayer_layers(l-2).Max_PoolLayer_obj.pre_conv_layer_n_kernel,ConvPoolLayer_layers(l-2).Max_PoolLayer_obj.s0,ConvPoolLayer_layers(l-2).Max_PoolLayer_obj.s1)
              }
              //把上次计算时的上一层=layer_input 赋值给上两层prev_layer_input
              prev_layer_input=Array.ofDim[Double](layer_input.length,layer_input(0).length,layer_input(0)(0).length)
              for(i1<-0 until layer_input.length){
                for(i2<-0 until layer_input(0).length){  
                  for(i3<-0 until layer_input(0)(0).length){
                    prev_layer_input(i1)(i2)(i3)=layer_input(i1)(i2)(i3)
                  }
                }
              }
              //上一层数据layer_input重新定义为ConvPoolLayer_layers(l-1).maxpool个单元
              layer_input = Array.ofDim[Double](ConvPoolLayer_layers(l-1).Max_PoolLayer_obj.pooled_input.length,ConvPoolLayer_layers(l-1).Max_PoolLayer_obj.pooled_input(0).length,ConvPoolLayer_layers(l-1).Max_PoolLayer_obj.pooled_input(0)(0).length)
              //根据prev_layer_input经过 sigmoid_layers(l-1) 后输出为 layer_input
              ConvPoolLayer_layers(l-1).cnn_forward(prev_layer_input)
              for(i1<-0 until ConvPoolLayer_layers(l-1).Max_PoolLayer_obj.pooled_input.length){
                for(i2<-0 until ConvPoolLayer_layers(l-1).Max_PoolLayer_obj.pooled_input(0).length){
                  for(i3<-0 until ConvPoolLayer_layers(l-1).Max_PoolLayer_obj.pooled_input(0)(0).length){
                    layer_input(i1)(i2)(i3)=binomial(1,ConvPoolLayer_layers(l-1).Max_PoolLayer_obj.pooled_input(i1)(i2)(i3),rng=new Random(1234) )//输出为0或者是1
                  }
                }
              }
            }
          }
          if(i<n_ConvPoolLayer){
            //使用layer_input 对第i层dA做训练
            ConvPooldALayer_layers(i).ConvLayer_obj.pre_train_da(layer_input,lr_chang, corruption_level,batch_num)
          }else{
            //输出层
            da_obj.train(x=flatten(layer_input),lr=lr_chang, corruption_level=corruption_level,batch_num=batch_num)
          }
        }
        if(i<n_ConvPoolLayer){
          println("cross_entropy="+ConvPooldALayer_layers(i).ConvLayer_obj.cross_entropy_result/batch_num) //debug
        }else{
          println("cross_entropy="+da_obj.cross_entropy_result/batch_num) //debug
        }
        //lr_chang=lr_chang*0.9
      }
      //保存第i层da的参数
      if(! (save_module_path=="")){
        if(i<n_ConvPoolLayer){
          ConvPooldALayer_layers(i).ConvLayer_obj.save_w(save_module_path+"cnn_mnist_module_da"+i+".txt") //debug 
        }else{
          da_obj.save_w(save_module_path+"cnn_mnist_module_mlp.txt")
        }
      }
    }
  }
}

object CNN_dA {
//预训练
def pre_train_mnist() {
    //读取训练集数据
    //val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/training_data.txt"
    val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/valid_data.txt"//train数据量大,暂时使用valid数据
    val width:Int=28;
    val height:Int=28;//debug 28*28
    val train_X:Array[Array[Array[Array[Double]]]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>{val tmp:Array[Array[Double]]=Array.ofDim[Double](height,width);for(i <- 0 until height){for(j <-0 until width){tmp(i)(j)=x._2(i*width+j)}};Array(tmp)})

    def trans_int_to_bin(int_in:Int):Array[Int]={
      val result:Array[Int]=Array(0,0,0,0,0,0,0,0,0,0);
      result(int_in)=1
      result
    }
    val train_Y:Array[Array[Int]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>trans_int_to_bin(x._1))
    val train_N: Int = train_X.length
    
    val rng: Random = new Random(123)
    var learning_rate: Double = 0.1
    val n_epochs: Int = 100
    //lenet5
    val classifier = new  CNN_dA(input_size=(height,width),output_size=10,n_kernel_Array=Array(6,16,120),kernel_size_Array=Array((5,5),(5,5),(4,4)),pool_size_Array=Array((2,2),(2,2),(1,1)),n_channel=1,n_hidden=84,_rng=null,activation="ReLU",activation_mlp="tanh")
    classifier.pretrain_sda_all(train_X=train_X, lr=learning_rate, corruption_level=0.1, epochs=n_epochs,batch_num_per=0.1, save_module_path="D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//") 
}  

//预训练后的cnn
def train_mnist_aft_pretrain() {
//读取训练集数据
    //val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/training_data.txt"
    val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/valid_data.txt"//train数据量大,暂时使用valid数据
    val width:Int=28;
    val height:Int=28;//debug 28*28
    val train_X:Array[Array[Array[Array[Double]]]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>{val tmp:Array[Array[Double]]=Array.ofDim[Double](height,width);for(i <- 0 until height){for(j <-0 until width){tmp(i)(j)=x._2(i*width+j)}};Array(tmp)})

    def trans_int_to_bin(int_in:Int):Array[Int]={
      val result:Array[Int]=Array(0,0,0,0,0,0,0,0,0,0);
      result(int_in)=1
      result
    }
    val train_Y:Array[Array[Int]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>trans_int_to_bin(x._1))
    val train_N: Int = train_X.length
    
    val rng: Random = new Random(123)
    var learning_rate: Double = 0.1
    val n_epochs: Int = 200
    
    //lenet5
    val classifier = new  CNN(input_size=(height,width),output_size=10,n_kernel_Array=Array(6,16,120),kernel_size_Array=Array((5,5),(5,5),(4,4)),pool_size_Array=Array((2,2),(2,2),(1,1)),n_channel=1,n_hidden=84,_rng=null,activation="ReLU",activation_mlp="tanh")
    for(i<-0 until classifier.n_ConvPoolLayer){
      classifier.ConvPoolLayer_layers(i).ConvLayer_obj.read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//cnn_mnist_module_da"+i+".txt")
    }
    classifier.mlp_layer.hidden_layers(0).read_w_module_from_da_rbm("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//cnn_mnist_module_mlp.txt")
    
    // train
    var epoch: Int = 0
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate, batch_num_per=0.01,alpha=0.1, save_module_path="",debug=false)
      learning_rate *=0.99
    } 
    
    /*
     * test
     * */
    val filePath_test:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt"  
    val test_X:Array[Array[Array[Array[Double]]]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>{val tmp:Array[Array[Double]]=Array.ofDim[Double](height,width);for(i <- 0 until height){for(j <-0 until width){tmp(i)(j)=x._2(i*width+j)}};Array(tmp)})
    val test_N: Int = 1000//test_X.length
    val test_Y_pred: Array[Array[Double]] = Array.ofDim[Double](test_N, 10)
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
    //test_CNN_simple()//ok
    //train_test_mnist()//--1层的ok
    pre_train_mnist()
    //train_mnist_aft_pretrain()
  }  

}