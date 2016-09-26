package dp_process_parallel

import scala.util.Random
import scala.math

import java.io.File; 
import java.io._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array


  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

class HiddenLayer_parallel( n_in_arg: Int, 
                            n_out_arg: Int, 
                            _W: Array[Array[Double]], 
                            _b: Array[Double], 
                            var rng: Random=null,
                            activation:String="sigmoid") {
  val n_in: Int=n_in_arg
  val n_out: Int=n_out_arg  
  if(rng == null) rng = new Random(1234)
  var a: Double = 0.0
  var W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)
  var W_add: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)
  var b: Array[Double] = new Array[Double](n_out)
  var b_add: Array[Double] = new Array[Double](n_out)
  var i: Int = 0
  init_w_module()
  if(_W == null) {
  } else {
    W = _W
  }

  if(_b != null) b = _b
  
  
  
  /*
   * 使用一个样本做 向前运算
   * */ 
  def forward(input: Array[Double]):Array[Double]={
    val result:ArrayBuffer[Double]=ArrayBuffer()
    for(i <-0 until n_out){
        result += output(input,W(i),b(i))
    } 
    result.toArray
  } 
  
  
  /*
   * 使用一个样本做向后传播
   * sum(x*w)=v-->active()-->y
   *   next_layer_n_in         下一层的n_in
   *   next_layer_n_out        下一层的n_out
   *   next_layer_w            下一层的W
   *   next_layer_dv           下一层的d_v 
   *   
   * */
  def backward(input:Array[Double],
                 next_layer_n_in:Int=0,
                 next_layer_n_out:Int=0,
                 next_layer_w:Array[Array[Double]]=null,
                 next_layer_dv:Array[Double]=null,             
                 dropout:Boolean=false,
                 mask:Array[Double]=Array()):(Array[Array[Double]],Array[Double],Array[Double])={
    //计算参数的更新大小    (最后输出)
    val W_add_tmp:Array[Array[Double]]=Array.ofDim[Double](n_out, n_in)//W的更新大小
    val b_add_tmp:Array[Double]=new Array[Double](n_out)//b的更新大小
    //本次bp的d_v
    val d_v:Array[Double]=new Array(n_out)
    //计算样本input经过本层后的线性输出值,在带入activation的导数中
    val output_d_s:ArrayBuffer[Double]=ArrayBuffer()
    for(i <-0 until n_out){
        output_d_s += doutput(input,W(i),b(i))
    }    
    //局部梯度 d_v=sum(下一层的d_v*下一层的w)*(样本input经过本层后的线性输出值,在带入activation的导数中)
    //不论下一层是logistic还是hidden都一样的逻辑
    for(j <- 0 until next_layer_n_in){
      //print(j+"\t"+next_layer_n_in+"\t"+d_v.length+"\t"+n_out+"\n")
      d_v(j)=0.0 //清空为0
      for(i <-0 until next_layer_n_out){
        d_v(j) += next_layer_w(i)(j)*next_layer_dv(i)  
      }
      d_v(j)= d_v(j)*output_d_s(j)
      if (dropout == true){
        d_v(j) = d_v(j)*mask(j)
      } 
      //遍历输入
      for(k <- 0 until n_in){
        W_add_tmp(j)(k) = input(k) * d_v(j)
      }
      b_add_tmp(j)=1.0*d_v(j)
    } 
    (W_add_tmp,b_add_tmp,d_v)
  }  
  
  /*
   * 函数
   * */
  //线性输出后经过activation
  def output(input: Array[Double], w: Array[Double], b: Double): Double = {
    var linear_output: Double = 0.0
    var j: Int = 0
    for(j <- 0 until n_in) {
      linear_output += w(j) * input(j)
    }
    linear_output += b
    if(activation=="sigmoid"){
      dp_process.utils.sigmoid(linear_output)  
    }else if(activation=="tanh"){
      dp_process.utils.tanh(linear_output)
    }else if(activation=="ReLU") {
      dp_process.utils.ReLU(linear_output)
    }else{
      0.0
    }
  }   
  //线性输出后经过activation的导数
  def doutput(input: Array[Double], w: Array[Double], b: Double): Double = {
    var linear_output: Double = 0.0
    var j: Int = 0
    for(j <- 0 until n_in) {
      linear_output += w(j) * input(j)
    }
    linear_output += b
    if(activation=="sigmoid"){
      dp_process.utils.dsigmoid(linear_output)  
    }else if(activation=="tanh"){
      dp_process.utils.dtanh(linear_output)
    }else if(activation=="ReLU") {
      dp_process.utils.dReLU(linear_output)
    }else{
      0.0
    }
  }
  //初始化清空系数w 
  def init_w_module():Unit={
    var i: Int = 0
    var j: Int = 0
    //val a: Double = 1 / n_in   //方案1 见yusugomori----未成功
    val a: Double = 4 * math.sqrt(6.0/(n_in + n_out))         //方案2 见lisa DeepLearningTutorials-master----对minst的sda和dbn很成功;cnn simple 以及train_test_mnist的单层
    //val a: Double =1/ math.pow(n_out,0.25)//用于dropout的两个实验ok  ok----relu,rl=0.1  learning_rate *=0.9  50times
    //(a^2/3)*m^0.5=1/9-->(a^4/9)*m=1/9-->a=1/m^0.25
    for(i <- 0 until n_out) 
      for(j <- 0 until n_in) 
        W(i)(j) = uniform(-a, a) 
    //则初始化b=0
    i= 0
    for(i <- 0 until n_out) b(i) = 0        
  }   
  //输出为0或者是1     计算完概率之后，根据当前概率来进行贝努利实验，得到0或1输出
  def sample_h_given_v(input: Array[Double], sample: Array[Double]) {
    var i: Int = 0
    for(i <- 0 until n_out) {
      sample(i) = binomial(1, output(input, W(i), b(i)))//输出为0或者是1
    }
  }  
  /*
   * 要随机地从n个数中以概率p对其进行选择，我们可以先生成一个掩膜（mask）
                 也即对这个n个数，分别以p进行确定其值为1（选中该值0)，以(1-p)确定其值为0（也即是未选中该值）
     python:mask = rng.binomial(size=input.shape,n=1,p=1-p)  # p is the prob of dropping
            mask = rng.binomial(size=5,n=1,p=0.8) =>[1,1,0,1,1]
   */
  def dropout(size:Int,p:Double,n:Int=1):Array[Double]={
    var result:ArrayBuffer[Double]=ArrayBuffer()
    for(i <-0 until size){
      result +=binomial(n,p) 
    }
    result.toArray
  } 
  
  def uniform(min: Double, max: Double): Double = {
    rng.nextDouble() * (max - min) + min
  }

  def binomial(n: Int, p: Double): Int = {
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
  
  //记录系数
  def save_w(file_module_in:String):Unit={
   val writer_module = new PrintWriter(new File(file_module_in)) 
   //write w
   writer_module.write(W.map(w_i=>w_i.mkString(sep=",")).mkString(sep="=sepi="))
   writer_module.write("\n")
   //write bias
   writer_module.write(b.mkString(sep=","))
   writer_module.write("\n")
  }
  
  //读取da rbm save_w输出的模型txt
  def read_w_module_from_da_rbm(file_module_in:String):Unit={
    var result:ArrayBuffer[String]=ArrayBuffer();
    Source.fromFile(file_module_in).getLines().foreach(line=>result+=line) 
    val result_w:Array[Array[Double]]=result(0).split("=sepi=").map(x=>x.split(",").map(y=>y.toDouble))
    for(i <- 0 until n_out)
      for(j <- 0 until n_in)
        W(i)(j) = result_w(i)(j)
    b=result(2).split(",").map(x=>x.toDouble)
  }    
  
}