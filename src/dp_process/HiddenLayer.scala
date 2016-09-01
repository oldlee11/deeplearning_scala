package dp_process

import scala.util.Random
import scala.math

import java.io.File; 
import java.io._
import scala.io.Source

import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

//默认使用sigmoid
class HiddenLayer(val n_in: Int, val n_out: Int, _W: Array[Array[Double]], _b: Array[Double], var rng: Random=null,val activation:String="sigmoid") {
  
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

  /*移至dp_process.utils
  def sigmoid(x: Double): Double = {
    return 1.0 / (1.0 + math.pow(math.E, -x))
  }*/

  if(rng == null) rng = new Random(1234)
  var a: Double = 0.0
  var W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)
  var b: Array[Double] = new Array[Double](n_out)
  var i: Int = 0
  init_w_module()
  if(_W == null) {
  } else {
    W = _W
  }

  if(_b != null) b = _b

  //add tanh  ReLU
  def output(input: Array[Double], w: Array[Double], b: Double): Double = {
    var linear_output: Double = 0.0
    var j: Int = 0
    for(j <- 0 until n_in) {
      linear_output += w(j) * input(j)
    }
    linear_output += b
    if(activation=="sigmoid"){
      utils.sigmoid(linear_output)  
    }else if(activation=="tanh"){
      utils.tanh(linear_output)
    }else if(activation=="ReLU") {
      utils.ReLU(linear_output)
    }else{
      0.0
    }
  } 
  
  //输出为0或者是1     计算完概率之后，根据当前概率来进行贝努利实验，得到0或1输出
  def sample_h_given_v(input: Array[Double], sample: Array[Double]) {
    var i: Int = 0
    for(i <- 0 until n_out) {
      sample(i) = binomial(1, output(input, W(i), b(i)))//输出为0或者是1
    }
  }
  
  //初始化清空系数w 
  def init_w_module():Unit={
    var i: Int = 0
    var j: Int = 0
    //val a: Double = 1 / n_in   //方案1 见yusugomori----未成功
    //val a: Double = 4 * math.sqrt(6.0/(n_in + n_out))         //方案2 见lisa DeepLearningTutorials-master----对minst的sda和dbn很成功
    val a: Double =1/ math.pow(n_out,0.25)//用于dropout的两个实验ok----relu,rl=0.1  learning_rate *=0.9  50times
    //(a^2/3)*m^0.5=1/9-->(a^4/9)*m=1/9-->a=1/m^0.25
    for(i <- 0 until n_out) 
      for(j <- 0 until n_in) 
        W(i)(j) = uniform(-a, a) 
    //则初始化b=0
    i= 0
    for(i <- 0 until n_out) b(i) = 0        
  }  

  
  ////////////////////////////new add:save////////////////////////////////////////////
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
  
  ////////////////////////////new add:bp////////////////////////////////////////////
  
  /*
   * 符号
   * sum(x*w)=v-->active()-->y
   *
   * */
  
  
  //在使用BP时计算的本层的局部梯度,即误差对v的偏导,如果不适用bp则不用使用
  val d_v:Array[Double]=new Array(n_out)
  var W_add: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)//初始化为0了
  var b_add: Array[Double] = new Array[Double](n_out)//初始化为0了
  
  //线性输出后经过activation的导出
  def doutput(input: Array[Double], w: Array[Double], b: Double): Double = {
    var linear_output: Double = 0.0
    var j: Int = 0
    for(j <- 0 until n_in) {
      linear_output += w(j) * input(j)
    }
    linear_output += b
    if(activation=="sigmoid"){
      utils.dsigmoid(linear_output)  
    }else if(activation=="tanh"){
      utils.dtanh(linear_output)
    }else if(activation=="ReLU") {
      utils.dReLU(linear_output)
    }else{
      0.0
    }
  } 
  
  //使用一个样本做 向前运算 
  def forward(input: Array[Double]):Array[Double]={
    val result:ArrayBuffer[Double]=ArrayBuffer()
    for(i <-0 until n_out){
        result += output(input,W(i),b(i))
    } 
    result.toArray
  } 

  //使用一个样本做 向后计算
  //next_layer 是下一层的网络  
  //下一层hidden_layers(i+1)是HiddenLayer
  //sum(x*w)=v-->active()-->y
  def backward_1(next_layer:HiddenLayer,input:Array[Double],batch_num:Int,lr:Double=0.1,L2_reg:Double=0.0,alpha:Double=0.9,dropout:Boolean=false,mask:Array[Double]=Array())={
    //计算样本input经过本层后的线性输出值,在带入activation的导数中
    var output_d_s:ArrayBuffer[Double]=ArrayBuffer()
    for(i <-0 until n_out){
        output_d_s += doutput(input,W(i),b(i))
    }     
    //局部梯度 d_v=sum(下一层的d_v*下一层的w)*(样本input经过本层后的线性输出值,在带入activation的导数中)
    for(j <- 0 until next_layer.n_in){
      d_v(j)=0.0 //清空为0
      for(i <-0 until next_layer.n_out){
        d_v(j)= d_v(j) + next_layer.W(i)(j)*next_layer.d_v(i)  
      }
      d_v(j)= d_v(j)*output_d_s(j)
      if (dropout == true){
        d_v(j) = d_v(j)*mask(j)
      } 
      //遍历输入
      for(k <- 0 until n_in){
        //原始：W(j)(k) +=lr*d_v(j)*input(k)/batch_num
        //第二版W(j)(k) +=(lr*d_v(j)*input(k)-lr*L2_reg*W(j)(k))/batch_num
        W_add(j)(k)=alpha*W_add(j)(k)+(lr*d_v(j)*input(k)-lr*L2_reg*W(j)(k))/batch_num
        W(j)(k) +=W_add(j)(k)
      }
      //第二版和原始版b(j) += lr*d_v(j)*1.0/batch_num//偏执节点相当于输入=1.0  
      b_add(j)=alpha*b_add(j)+lr*d_v(j)*1.0/batch_num
      b(j) +=b_add(j)
    } 
  }
  
  //使用一个样本做 向后计算
  //next_layer 是下一层的网络  
  //下一层hidden_layers(i+1)是LogisticRegression
  //sum(x*w)=v-->active()-->y
  def backward_2(next_layer:LogisticRegression,input:Array[Double],batch_num:Int,lr:Double=0.1,L2_reg:Double=0.0,alpha:Double=0.9,dropout:Boolean=false,mask:Array[Double]=Array())={
    //计算样本input经过本层后的线性输出值,在带入activation的导数中
    var output_d_s:ArrayBuffer[Double]=ArrayBuffer()
    for(i <-0 until n_out){
        output_d_s += doutput(input,W(i),b(i))
    }     
    //局部梯度 d_v=sum(下一层的d_v*下一层的w)*(样本input经过本层后的线性输出值,在带入activation的导数中)
    for(j <- 0 until next_layer.n_in){
      d_v(j)=0.0 //清空为0
      for(i <-0 until next_layer.n_out){
        d_v(j)= d_v(j) + next_layer.W(i)(j)*next_layer.d_y(i)  
      }
      d_v(j)= d_v(j)*output_d_s(j)
      if (dropout == true){
        d_v(j) = d_v(j)*mask(j)
      } 
      //遍历输入
      for(k <- 0 until n_in){
        //原始：W(j)(k) +=lr*d_v(j)*input(k)/batch_num
        //第二版W(j)(k) +=(lr*d_v(j)*input(k)-lr*L2_reg*W(j)(k))/batch_num
        W_add(j)(k)=alpha*W_add(j)(k)+(lr*d_v(j)*input(k)-lr*L2_reg*W(j)(k))/batch_num
        W(j)(k) +=W_add(j)(k)
      }
      //第二版和原始版b(j) += lr*d_v(j)*1.0/batch_num//偏执节点相当于输入=1.0  
      b_add(j)=alpha*b_add(j)+lr*d_v(j)*1.0/batch_num
      b(j) +=b_add(j)    
    }     
  }  
  
  
  //要随机地从n个数中以概率p对其进行选择，我们可以先生成一个掩膜（mask）
  //也即对这个n个数，分别以p进行确定其值为1（选中该值0)，以(1-p)确定其值为0（也即是未选中该值）
  //python:mask = rng.binomial(size=input.shape,n=1,p=1-p)  # p is the prob of dropping
  //       mask = rng.binomial(size=5,n=1,p=0.8) =>[1,1,0,1,1]
  def dropout(size:Int,p:Double,n:Int=1):Array[Double]={
    var result:ArrayBuffer[Double]=ArrayBuffer()
    for(i <-0 until size){
      result +=binomial(n,p) 
    }
    result.toArray
  }
  
}  
