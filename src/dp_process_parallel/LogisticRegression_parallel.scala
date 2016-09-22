package dp_process_parallel

import scala.math
import scala.math.log

import java.io.File; 
import java.io._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

class LogisticRegression_parallel(n_in_arg: Int,n_out_arg: Int) {
  val n_in: Int=n_in_arg
  val n_out: Int=n_out_arg
  val W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)//初始化为了0.0
  val W_add: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)//初始化为了0.0
  val b: Array[Double] = new Array[Double](n_out)//初始化为了0.0
  val b_add: Array[Double] = new Array[Double](n_out)//初始化为了0.0
  
  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)
    x:一个样本的x数据,据取值为[0,1]之间
    y:取值为例如Array(0,0,1,0,0,0)表示分类2
            输出为(W_add_tmp,b_add_tmp,cross_entropy_result)
    */  
  def train(x: Array[Double], 
            y: Array[Int]):(Array[Array[Double]],Array[Double],Array[Double],Double)= {
    val W_add_tmp: Array[Array[Double]]=Array.ofDim[Double](n_out, n_in)
    val b_add_tmp: Array[Double] = new Array[Double](n_out)
    val d_y: Array[Double] = new Array[Double](n_out)//局部梯度
    val p_y_given_x: Array[Double] = new Array[Double](n_out)
    for(i <- 0 until n_out) {
      p_y_given_x(i) = 0
      for(j <- 0 until n_in) {
        p_y_given_x(i) += W(i)(j) * x(j)
      }
      p_y_given_x(i) += b(i)
    }
    val p_y_given_x_softmax:Array[Double]=softmax(p_y_given_x)//相当于p_y_given_x_softmax=softmax(p_y_given_x)
    val cross_entropy_result:Double=cross_entropy(y.map(x=>x.toDouble),p_y_given_x_softmax)
    for(i <- 0 until n_out) {
      d_y(i) = y(i) - p_y_given_x_softmax(i)
      for(j <- 0 until n_in) {
        W_add_tmp(i)(j)= d_y(i) * x(j)
      }
      b_add_tmp(i)=d_y(i)*1.0
    }
    (W_add_tmp,b_add_tmp,d_y,cross_entropy_result)
  }  
  
  /*
   * 对一个样本做预测，就是向前传播
   * */
  def predict(x: Array[Double]):Array[Double]={
    var i: Int = 0
    var j: Int = 0
    val tmp:Array[Double]=new Array(n_out)
    for(i <- 0 until n_out) {
      tmp(i) = 0
      for(j <- 0 until n_in) {
        tmp(i) += W(i)(j) * x(j)
      }
      tmp(i) += b(i)
    }
    softmax(tmp)
  }
  
  /*
   * 函数
   * */
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
  //使用交叉信息嫡cross-entropy衡量样本输入和经过编解码后输出的相近程度
  //值在>=0 当为0时 表示距离接近
  //每次批量迭代后该值越来越小
  def cross_entropy(x: Array[Double], z: Array[Double]):Double={
   -1.0* (0 until x.length).map(i=>x(i)*log(z(i))+(1-x(i))*log(1-z(i))).reduce(_+_)
  }
  
}