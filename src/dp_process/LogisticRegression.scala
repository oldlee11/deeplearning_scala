package dp_process

import scala.math
import scala.math.log

import java.io.File; 
import java.io._
import scala.io.Source

import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

class LogisticRegression(val n_in: Int, val n_out: Int) {
  val d_y: Array[Double] = new Array[Double](n_out)//局部梯度
  
  val W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)//初始化为了0.0
  val b: Array[Double] = new Array[Double](n_out)//初始化为了0.0
  var W_add: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)//初始化为0了
  var b_add: Array[Double] = new Array[Double](n_out)//初始化为0了  
  var cross_entropy_result:Double=0.0
  
  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)
    x:一个样本的x数据,据取值为[0,1]之间
    y:取值为例如Array(0,0,1,0,0,0)表示分类2
    lr:参数迭代的学习率
    batch_num 是一次批量迭代使用的样本数,由于每一次训练时,样本仅输入1个,参数(W,vbias,hbias)迭代计算时：W(t+1)=W(t)+lr*(局部误差/batch_num)
                                                 即每一次迭代仅仅更新1/batch_num倍,如果batch_num=训练集样本个数,则当所有训练集都迭代一遍后则 为 W(t+1)=W(t)+lr*(局部误差)
                                                如果batch_num=1则表示每个样本迭代一次 W(t+1)=W(t)+lr*(局部误差) 即不是批量迭代而是随机迭代了  
  */  
  def train(x: Array[Double], y: Array[Int], lr: Double,batch_num:Int,L2_reg:Double=0.0,alpha:Double=0.0) {
    val p_y_given_x: Array[Double] = new Array[Double](n_out)
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_out) {
      p_y_given_x(i) = 0
      for(j <- 0 until n_in) {
        p_y_given_x(i) += W(i)(j) * x(j)
      }
      p_y_given_x(i) += b(i)
    }
    val p_y_given_x_softmax:Array[Double]=softmax(p_y_given_x)//相当于p_y_given_x_softmax=softmax(p_y_given_x)
    this.cross_entropy_result=this.cross_entropy_result + cross_entropy(y.map(x=>x.toDouble),p_y_given_x_softmax)
    for(i <- 0 until n_out) {
      d_y(i) = y(i) - p_y_given_x_softmax(i)
      for(j <- 0 until n_in) {
        //原始版W(i)(j) += lr * d_y(i) * x(j) / batch_num
        //第二版W(i)(j) += (lr *d_y(i)*x(j) -lr*L2_reg*W(i)(j))/ batch_num
        W_add(i)(j)=alpha*W_add(i)(j)+ (lr *d_y(i)*x(j) -lr*L2_reg*W(i)(j))/ batch_num
        W(i)(j) +=W_add(i)(j)
      }
      //原始和第二版b(i) += lr * d_y(i) / batch_num
      b_add(i)=alpha*b_add(i)+ lr * d_y(i) / batch_num
      b(i) +=b_add(i)
    }
  }

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
  
  //使用交叉信息嫡cross-entropy衡量样本输入和经过编解码后输出的相近程度
  //值在>=0 当为0时 表示距离接近
  //每次批量迭代后该值越来越小
  def cross_entropy(x: Array[Double], z: Array[Double]):Double={
   -1.0* (0 until x.length).map(i=>x(i)*log(z(i))+(1-x(i))*log(1-z(i))).reduce(_+_)
  }
    
  
  //记录系数
  def save_w(file_module_in:String):Unit={
   val writer_module = new PrintWriter(new File(file_module_in)) 
   //write w
   writer_module.write(W.map(w_i=>w_i.mkString(sep=",")).mkString(sep="=sepi="))
   writer_module.write("\n")
   //write b
   writer_module.write(b.mkString(sep=","))
   writer_module.write("\n")
   writer_module.close()
  }
  
  //读取 save_w输出的模型txt
  //并修改LR的W    b
  def read_w_module(file_module_in:String):Unit={
    var result:ArrayBuffer[String]=ArrayBuffer();
    Source.fromFile(file_module_in).getLines().foreach(line=>result+=line) 
    //W
    val result_w:Array[Array[Double]]=result(0).split("=sepi=").map(x=>x.split(",").map(y=>y.toDouble))
    for(i <- 0 until n_out)
      for(j <- 0 until n_in)
        W(i)(j) = result_w(i)(j)
    //b
    val tmp:Array[String]=result(1).split(",")
    for(i <- 0 until n_out){
      b(i)=tmp(i).toDouble
    }
  }
  
  //初始化均为0???????
  def init_w_module(){
    //w
    for(i <- 0 until n_out)
      for(j <- 0 until n_in)
        W(i)(j) = 0.0
    //b
    for(i <- 0 until n_out){
      b(i)=0.0
    }    
  } 
  
}

object LogisticRegression {
  def test_LR_simple() {
    var learning_rate: Double = 0.1
    val n_epochs: Int = 500
    val train_N: Int = 9
    val test_N: Int = 3
    val n_in: Int = 9
    val n_out: Int = 3
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
    // construct
    val classifier = new LogisticRegression(n_in, n_out)
    // train
    var epoch: Int = 0
    var i: Int = 0
    for(epoch <- 0 until n_epochs) {
      for(i <- 0 until train_N) {
        classifier.train(train_X(i), train_Y(i), learning_rate,train_N)
        //classifier.train(train_X(i), train_Y(i), learning_rate,train_N,alpha=0.9)
        //classifier.train(train_X(i), train_Y(i), learning_rate,train_N,L2_reg=0.1)
      }
      //learning_rate *= 0.95
    }
    // test data
    val test_X: Array[Array[Double]] = Array(
      Array(1, 0, 1, 0, 0, 0,0,0,0),
      Array(0, 0, 1, 1, 1, 0,0,0,0),
      Array(0, 0, 0, 0, 0, 0,1,1,1)
    )
    val test_Y: Array[Array[Double]] = Array.ofDim[Double](test_N, n_out)
    // test
    var j: Int = 0
    for(i <- 0 until test_N) {
      test_Y(i)=classifier.predict(test_X(i))
      for(j <- 0 until n_out) {
        printf("%.5f ", test_Y(i)(j))
      }
      println()
      /*
       * 0.92048 0.04735 0.03217 
       * 0.01120 0.98404 0.00475 
       * 0.00196 0.00374 0.99430 
       * */
    }
    //保存系数
    classifier.save_w("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//LR_simple_module.txt")
    //清空系数
    classifier.init_w_module()
    //读取系数
    classifier.read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//LR_simple_module.txt")
    //再次保存系数
    classifier.save_w("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//LR_simple_module2.txt")
  }
  
  def main(args: Array[String]) {
    test_LR_simple()
  }

}
