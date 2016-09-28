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

/* 实现一个列内部的一个hidden
 * 
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
 * */

class RNN_HiddenLayer(n_in:Int,
                      n_out:Int,
                      _W: Array[Array[Double]]=null, 
                      _b: Array[Double]=null, 
                      _rng: Random=null,
                      activation:String="sigmoid") extends HiddenLayer(
                      n_in,
                      n_out, 
                      _W, 
                      _b, 
                      _rng,
                      activation){
  //使用一个样本做 向前运算 
  override def forward(input: Array[Double]):Array[Double]={
    /* HiddenLayer
    val result:ArrayBuffer[Double]=ArrayBuffer()
    for(i <-0 until n_out){
        result += output(input,W(i),b(i))
    } 
    result.toArray
    */
    val result:Array[Double]=new Array(n_out)
    for(i <-0 until n_out){
        result(i) = output(input,W(i),b(i))
    } 
    result
  } 

  /*
   * input 是上一层的输入=Hidden
   * pre_time_hidden_input 是本层的上一个时间状态数值,RNN独有
   * w,b 同HIdden 是上一层到本层的系数
   * w_hh,b_hh是 本层的上一个时间状态到本层本时间的系数,RNN独有
   * */
  def RNN_output(input:Array[Double],
                      pre_time_hidden_input:Array[Double],
                      w:Array[Double],
                      b:Double,
                      w_hh:Array[Double],
                      b_hh:Double): Double = {
    /* HiddenLayer的output()
     * =activation(w*input+b)
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
    */
    //RNN_output()=activation(w*input+b+w_hh*pre_time_hidden_input+b_hh)
    var linear_output: Double = 0.0
    var j: Int = 0
    for(j <- 0 until n_in) {
      linear_output += w(j) * input(j)
    }
    for(j<- 0 until n_out){
      linear_output += w_hh(j) * pre_time_hidden_input(j)
    }
    linear_output += (b+b_hh)
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
  
}  

object RNN_HiddenLayer {
  
}

