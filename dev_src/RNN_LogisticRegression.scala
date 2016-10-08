package dp_process

import scala.math
import scala.math.log

  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

//和LogisticRegression相比 只是把train拆解为了向前和向后过程
//并且 不做参数更新 仅做梯度的计算
class RNN_LogisticRegression(val n_in: Int, 
                             val n_out: Int) {
  val W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)//初始化为了0.0
  val b: Array[Double] = new Array[Double](n_out)//初始化为了0.0
  
  
  /* **********************************
   * *********      向前传播              **********
   * **********************************/ 
  //x 是一个样本的x
  def forward(x: Array[Double]):Array[Double]={
    val p_y_given_x: Array[Double] = new Array[Double](n_out)
    for(i <- 0 until n_out) {
      p_y_given_x(i) = 0
      for(j <- 0 until n_in) {
        p_y_given_x(i) += W(i)(j) * x(j)
      }
      p_y_given_x(i) += b(i)
    }
    softmax(p_y_given_x)
  }
  
  
  /* **********************************
   * *********      向后传播              **********
   * **********************************/
  //y是一个样本的y
  //x是一个样本的x
  //p_y_given_x_softmax是forward的输出
  //输出(W_add_tmp,b_add_tmp,d_y,cross_entrpy)
  def backward(p_y_given_x_softmax:Array[Double],
               x:Array[Double],
               y:Array[Int]):(Array[Array[Double]],Array[Double],Array[Double],Double)={
    val d_y: Array[Double] = new Array[Double](n_out)//局部梯度 用于后续的网络做bp
    val W_add_tmp: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)//W的更新大小
    val b_add_tmp: Array[Double] = new Array[Double](n_out)//b的更新大小
    for(i <- 0 until n_out) {
      d_y(i) = y(i) - p_y_given_x_softmax(i)
      for(j <- 0 until n_in) {
        W_add_tmp(i)(j)=d_y(i)*x(j)
      }
      b_add_tmp(i)=d_y(i)
    }
    (W_add_tmp,b_add_tmp,d_y,cross_entropy(y.map(x=>x.toDouble),p_y_given_x_softmax))
  }
  
  /* **********************************
   * *********      函数                       **********
   * **********************************/ 
  def softmax(x: Array[Double]):Array[Double]= {
    val result:Array[Double]=new Array(n_out)
    var max: Double = 0.0
    var sum: Double = 0.0
    var i: Int = 0
    for(i <- 0 until n_out) if(max < x(i)) max = x(i)
    for(i <- 0 until n_out) {
      result(i) = math.exp(x(i) - max)
      sum += result(i)
    }
    for(i <- 0 until n_out) result(i)=result(i)/sum
    result
  }
  def predict(x: Array[Double]):Array[Double]={
    forward(x)
  }  
  //使用交叉信息嫡cross-entropy衡量样本输入和经过编解码后输出的相近程度
  //值在>=0 当为0时 表示距离接近
  //每次批量迭代后该值越来越小
  def cross_entropy(x: Array[Double], z: Array[Double]):Double={
   -1.0* (0 until x.length).map(i=>x(i)*log(z(i))+(1-x(i))*log(1-z(i))).reduce(_+_)
  }
  
}