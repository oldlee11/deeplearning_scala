package dp_process

import scala.util.Random
import scala.math

import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

/* 
 * 实现一个列内部的一个hidden
 * 内部主要是函数,w和b等系数没有存于内部(由于权值共享)
 * 可以做批量梯度下降(类似于HiddenLayer_parallel)
 */
 
/* 
 * 大致示意图-1(每个时刻的mlp均有输出)
 * 
 *                   |        MLP(k-1)         |        MLP(k-1)        ... ...         MLP(last)       |  
 *                   |                         |                         |   |                          | 
 *                   | logisticregression_layer| logisticregression_layer|   |  logisticregression_layer| 
 *                   |          /\             |          /\             |   |           /\             |
 *                   |          ||             |          ||             | . |           ||             |                  
 *                   |     hidden_layer[last] ====>  hidden_layer[last] ===.======> hidden_layer[last]  |
 *                   |          /\             |          /\             | . |           /\             |
 *                   |          ||             |          ||             |   |           ||             |
 *                   |        ... ...          |        ... ...          | . |         ... ...          |
 *                   |     hidden_layer[i] ========> hidden_layer[i] ======.======> hidden_layer[i]     |
 *                   |          /\             |          /\             | . |           /\             |
 *                   |          ||             |          ||             |   |           ||             |
 *                   |        ... ...          |        ... ...          | . |         ... ...          |
 *                   |     hidden_layer[0] ========> hidden_layer[0] ======.======> hidden_layer[0]     |
 *                   |          /\             |          /\             | . |           /\             |
 *                   |          ||             |          ||             |   |           ||             |
 *                   |        inputs           |        inputs           |   |         inputs           |
 *
 * 
 * 
 * 
 * 大致示意图-2(只在最后时刻的mlp有输出)
 * 
 *                   |        MLP(k-1)         |        MLP(k-1)        ... ...         MLP(last)       |  
 *                   |                         |                         |   |                          | 
 *                   |                         |                         |   |  logisticregression_layer| 
 *                   |                         |                         |   |           /\             |
 *                   |                         |                         | . |           ||             |                  
 *                   |     hidden_layer[last] ====>  hidden_layer[last] ===.======> hidden_layer[last]  |
 *                   |          /\             |          /\             | . |           /\             |
 *                   |          ||             |          ||             |   |           ||             |
 *                   |        ... ...          |        ... ...          | . |         ... ...          |
 *                   |     hidden_layer[i] ========> hidden_layer[i] ======.======> hidden_layer[i]     |
 *                   |          /\             |          /\             | . |           /\             |
 *                   |          ||             |          ||             |   |           ||             |
 *                   |        ... ...          |        ... ...          | . |         ... ...          |
 *                   |     hidden_layer[0] ========> hidden_layer[0] ======.======> hidden_layer[0]     |
 *                   |          /\             |          /\             | . |           /\             |
 *                   |          ||             |          ||             |   |           ||             |
 *                   |        inputs           |        inputs           |   |         inputs           |  
 *                   
 *                   
 *                   
 *      
 *                   
 * 相似示意图(带参数说明)
 * 不同时间状态下的mlp内,每个层的w和b以及w_hh,b_hh权值共享,输出层没有递归
 * 
 *                   |<------------------------  win_times -------------------------------------------- >|
 * |                 |       MLP(k-1)    |              MLP(k-1)                      |    MLP(k-1)      | 
 * |                 |       t=k-1       |              t=k                           |    t=k+1         |
 * |                 |                   |                                            |                  |
 * | lr_layer----------------------------------logisticregression_layer(n_out个神经元)   |                  |                 
 * |                 |                   |              /\                            |                  |
 * |                 |                   |              ||w_lr                        |                  |
 * |                 |                   |              ||b_lr                        |                  |
 * |                 |                   |           ... ...                          |                  | 
 * |                 |           w_hidden2hidden[i]     ||                       w_hidden2hidden[i]      |
 * | hidden_layer[i]------------==================> hidden_layer[i](nhidden个神经元)==================>     |
 * |                 |           b_hidden2hidden[i]     /\                       b_hidden2hidden[i]      |
 * |                 |                   |              ||w_hidden[i]                 |                  |
 * |                 |                   |              ||b_hidden[i]                 |                  |
 * |                 |                   |           ... ...                          |                  |
 * |                 |           w_hidden2hidden[0]     ||                       w_hidden2hidden[0]      |
 * | hidden_layer[0]------------==================> hidden_layer[0](nhidden个神经元)==================>     |
 * |                 |           b_hidden2hidden[0]     /\                       b_hidden2hidden[0]      |
 * |                 |                   |              ||w_hidden[0]                 |                  |
 * |                 |                   |              ||b_hidden[0]                 |                  |
 * |                 |                   |          input(n_in个神经元)                  |                  |
 * 
 * */

class RNN_HiddenLayer(_n_in:Int,
                      _n_out:Int,
                      _W: Array[Array[Double]]=null, 
                      _b: Array[Double]=null,
                      _rng: Random=null,
                      activation:String="sigmoid") {
  
  val rng=if(_rng==null) new Random(1234) else _rng
  val n_in:Int=_n_in
  val n_out:Int=_n_out
  var W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)     //相当于图中的w_hidden[i]
  var b: Array[Double] = new Array[Double](n_out)                    //相当于图中的b_hidden[i]
  var W_hh: Array[Array[Double]] = Array.ofDim[Double](n_out, n_out) //相当于图中的w_hidden2hidden[i]
  var b_hh: Array[Double] = new Array[Double](n_out)                 //相当于图中的b_hidden2hidden[i]
  init_w_module()
  init_whh_module()
  if(_W == null) {
  } else {
    W = _W
  }
  if(_b != null) b = _b  
  
  
  /* **********************************
   * *********      向前传播              **********
   * **********************************/ 
  /*  使用一个样本做 向前运算
   * 输入
   * input 是上一层的输入=HiddenLayer
   * pre_time_hidden_input 是本层的上一个时间状态数值 =null 表示第一层,即每没有前面一层
   * 如果hiddenlayer是第i层 则
   * W_arg    相当于w_hidden[i]
   * b_arg    相当于b_hidden[i]
   * W_hh_arg 相当于w_hidden2hidden[i]
   * b_hh_arg 相当于b_hidden2hidden[i]
   * 输出
   * (正向传播输出output,用于反向传播的doutput)
   * output=activation_fun(line_out)
   * doutput=dactivation_fun(line_out)
   * */  
  def forward(input:Array[Double],
              pre_time_hidden_input:Array[Double]=null,
              W_arg:Array[Array[Double]],
              b_arg:Array[Double],
              W_hh_arg:Array[Array[Double]],
              b_hh_arg:Array[Double]):(Array[Double],Array[Double])={
    val result_output:Array[Double]=new Array(n_out)//用于正向传播=activation_fun(line_out)
    val result_doutput:Array[Double]=new Array(n_out)//用于反向传播=dactivation_fun(line_out)
    /* HiddenLayer的output()
     * =activation(w*input+b)
     * RNN_output()
     * =activation(w*input+b+w_hh*pre_time_hidden_input+b_hh)
     */
    var linear_output_tmp: Double=0.0
    for(i <-0 until n_out){
      linear_output_tmp = 0.0
      //计算w*input+b 同HiddenLayer
      for(j <- 0 until n_in) {
        linear_output_tmp += W_arg(i)(j) * input(j)
      }
      linear_output_tmp += b_arg(i)
      if(pre_time_hidden_input==null){
        //如果null 则表示是第一层,即没有更前面的hidden
      }else{
        //计算w_hh*pre_time_hidden_input+b_hh RNN独有
        for(j<- 0 until n_out){
          linear_output_tmp += W_hh_arg(i)(j) * pre_time_hidden_input(j)
        }
        linear_output_tmp += b_hh_arg(i)
      }
      result_output(i)=activation_fun(linear_output_tmp)
      result_doutput(i)=dactivation_fun(linear_output_tmp)
    } 
    (result_output,result_doutput)
  } 
  
  
  /* **********************************
   * *********      向后传播              **********
   * **********************************/  
  /*  bptt方式 使用一个样本做 向后运算
   *  输入
   *   input 是上一层的输入=HiddenLayer
   *   pre_time_hidden_input 是本层的上一个时间状态数值
   *   doutput                 向前forward的输出值
   *   next_layer_w            下一层的W    相当于w_hidden[i+1]  如果等于null表示没有logistics输出层
   *   next_layer_dv           下一层的d_v                     如果等于null表示没有logistics输出层
   *   next_time_w             同一层的下一个时间状态的hidden的W  相当于w_hidden2hidden[i](权值是共享的)
   *   next_time_dv            同一层的下一个时间状态的hidden的dv
   *  输出
   *   (W_add_tmp,b_add_tmp,W_hh_add_tmp,b_hh_add_tmp,d_v)
   * */    
  def backward(input:Array[Double],
               pre_time_hidden_input:Array[Double],
               doutput:Array[Double],
               next_time_w:Array[Array[Double]],
               next_time_dv:Array[Double],
               next_layer_w:Array[Array[Double]]=null,
               next_layer_dv:Array[Double]=null,                    
               dropout:Boolean=false,
               mask:Array[Double]=Array()):(Array[Array[Double]],Array[Double],Array[Array[Double]],Array[Double],Array[Double])={
    //本次bp的d_v 用于上一层的反向传播 以及同一层的上一个时间状态的反向传播
    val d_v:Array[Double]=new Array(n_out)    
    //计算参数的更新大小    (最后输出)
    val W_add_tmp:Array[Array[Double]]=Array.ofDim[Double](n_out, n_in)//W的更新大小
    val b_add_tmp:Array[Double]=new Array[Double](n_out)//b的更新大小
    val W_hh_add_tmp:Array[Array[Double]]=Array.ofDim[Double](n_out, n_in)//W的更新大小
    val b_hh_add_tmp:Array[Double]=new Array[Double](n_out)//b的更新大小    
    //局部梯度 d_v=sum(下一层的d_v*下一层的w)*(样本input经过本层后的线性输出值,在带入activation的导数中)
    //不论下一层是logistic还是hidden都一样的逻辑
    val next_layer_n_out:Int=next_layer_w.length
    val next_layer_n_in:Int =next_layer_w(0).length
    for(j <- 0 until n_out){
      /*
       * 计算dv
       * 根据next_layer的dv 计算本层dv的nextlayer部分
       * 根据next_time的dv 计算本层dv的nexttime部分
       * 本层的dv=nextlayer的反向dv+nexttime的反向dv
       * */
      d_v(j)=0.0 //清空为0
      if(next_layer_w == null || next_layer_dv==null){
        //如果next_layer的w或dv有一个为null 则不计算 表示没有输出层
      }else{
        //计算nextlayer部分的dv
        for(i <-0 until next_layer_n_out){
          d_v(j) += next_layer_w(i)(j)*next_layer_dv(i)  
        }
      }
      //计算nexttime部分的dv
      for(i <-0 until n_out){
        d_v(j) += next_time_w(i)(j)*next_time_dv(i)  
      }      
      d_v(j)= d_v(j)*doutput(j)
      //droupout
      if (dropout == true){
        d_v(j) = d_v(j)*mask(j)
      } 
      /*
       * 计算add_tmp
       * */
      for(k <- 0 until n_in){
        W_add_tmp(j)(k) = input(k) * d_v(j)
      }
      b_add_tmp(j)=1.0*d_v(j)
      for(k <- 0 until n_out){
        W_hh_add_tmp(j)(k) = pre_time_hidden_input(k) * d_v(j)
      }
      b_hh_add_tmp(j)=1.0*d_v(j)      
    } 
    (W_add_tmp,b_add_tmp,W_hh_add_tmp,b_hh_add_tmp,d_v)
  }
  
 
  /* **********************************
   * *********      函数                       **********
   * **********************************/  
  def activation_fun(linear_output:Double):Double={
      if(activation=="sigmoid"){
        utils.sigmoid(linear_output)  
      }else if(activation=="tanh"){
        utils.tanh(linear_output)
      }else if(activation=="ReLU") {
        utils.ReLU(linear_output)
      }else if(activation=="linear"){
        linear_output
      }else{
        0.0
      }
  }
  def dactivation_fun(linear_output:Double):Double={
      if(activation=="sigmoid"){
        utils.dsigmoid(linear_output)  
      }else if(activation=="tanh"){
        utils.dtanh(linear_output)
      }else if(activation=="ReLU") {
        utils.dReLU(linear_output)
      }else if(activation=="linear"){
        1.0
      }else{
        0.0
      }
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
  //初始化清空系数w 
  def init_w_module():Unit={
    val a: Double = 4 * math.sqrt(6.0/(n_in + n_out))
    for(i <- 0 until n_out) 
      for(j <- 0 until n_in) 
        W(i)(j) = uniform(-a, a) 
    for(i <- 0 until n_out) b(i) = 0        
  }    
  def init_whh_module():Unit={
    val a: Double = 4 * math.sqrt(6.0/(n_out + n_out))
    for(i <- 0 until n_out) 
      for(j <- 0 until n_out) 
        W_hh(i)(j) = uniform(-a, a) 
    for(i <- 0 until n_out) b_hh(i) = 0        
  }   
  
}  

object RNN_HiddenLayer {
  
}

