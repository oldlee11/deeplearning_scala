package dp_process_breeze

import scala.util.Random
import scala.math.log

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._
import breeze.numerics._
  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

class LogisticRegression_breeze(n_in_arg:Int,n_out_arg:Int) {

  /**********************
   ****   init       ****
   **********************/  
  val n_in: Int=n_in_arg
  val n_out: Int=n_out_arg
  val W: DenseMatrix[Double] =matrix_utils.build_2d_matrix((n_out, n_in))//初始化为了0.0
  val W_add: DenseMatrix[Double] =matrix_utils.build_2d_matrix((n_out, n_in))//初始化为了0.0
  val b: DenseVector[Double] = matrix_utils.build_1d_matrix(n_out)//初始化为了0.0
  val b_add: DenseVector[Double] = matrix_utils.build_1d_matrix(n_out)//初始化为了0.0  
  
  
  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)
    x:一个样本的x数据,据取值为[0,1]之间
    y:取值为例如Array(0,0,1,0,0,0)表示分类2
            输出为(W_add_tmp,b_add_tmp,cross_entropy_result)
    */  
  def train(x: DenseVector[Double], 
            y: DenseVector[Double]):(DenseMatrix[Double],DenseVector[Double],DenseVector[Double],Double)= {
    /*
    val p_y_given_x: Array[Double] = new Array[Double](n_out)
    for(i <- 0 until n_out) {
      p_y_given_x(i) = 0
      for(j <- 0 until n_in) {
        p_y_given_x(i) += W(i)(j) * x(j)
      }
      p_y_given_x(i) += b(i)
    }*/
    print(W.rows+"\t"+W.cols+"\t"+x.length+"\t"+b.length+"\n")
    val p_y_given_x: DenseVector[Double] =(W*x)+b
    val p_y_given_x_softmax:DenseVector[Double]=softmax(p_y_given_x)//相当于p_y_given_x_softmax=softmax(p_y_given_x)
    val cross_entropy_result:Double=cross_entropy(y,p_y_given_x_softmax)
    /*
    val W_add_tmp: Array[Array[Double]]=Array.ofDim[Double](n_out, n_in)
    val b_add_tmp: Array[Double] = new Array[Double](n_out)
    val d_y: Array[Double] = new Array[Double](n_out)//局部梯度
    for(i <- 0 until n_out) {
      d_y(i) = y(i) - p_y_given_x_softmax(i)
      for(j <- 0 until n_in) {
        W_add_tmp(i)(j)= d_y(i) * x(j)
      }
      b_add_tmp(i)=d_y(i)*1.0
    }
    */
    val d_y: DenseVector[Double] =y - p_y_given_x_softmax//局部梯度    
    val W_add_tmp: DenseMatrix[Double]=d_y*x.t
    val b_add_tmp: DenseVector[Double] = d_y
    (W_add_tmp,b_add_tmp,d_y,cross_entropy_result)
  }  
  
  /*
   * 对一个样本做预测，就是向前传播
   * */
  def predict(x: DenseVector[Double]):DenseVector[Double]={
    /*
    val tmp:Array[Double]=new Array(n_out)
    for(i <- 0 until n_out) {
      tmp(i) = 0
      for(j <- 0 until n_in) {
        tmp(i) += W(i)(j) * x(j)
      }
      tmp(i) += b(i)
    }
    softmax(tmp)
    */
    softmax(W*x+b)
  }
  
  /*
   * 函数
   * */
  def softmax(x: DenseVector[Double]):DenseVector[Double]= {
    /* 计算x的最大值
     * for(i <- 0 until n_out) if(max < x(i)) max = x(i)
     * */
    var max: Double = breeze.linalg.max(x) 
    /* 对x内么个元素做math.exp(x_i - max)处理
    for(i <- 0 until n_out) {
      result(i) = math.exp(x(i) - max)
      sum += result(i)
    }*/
    val result:DenseVector[Double]=x.map(x_i=>math.exp(x_i - max))
    var sum: Double = breeze.linalg.sum(result)
    /* result内的每个值除以result的最大值
    for(i <- 0 until n_out) result(i)=result(i)/sum
    */
    x:/sum
  }
  //使用交叉信息嫡cross-entropy衡量样本输入和经过编解码后输出的相近程度
  //值在>=0 当为0时 表示距离接近
  //每次批量迭代后该值越来越小
  def cross_entropy(x: DenseVector[Double], z: DenseVector[Double]):Double={
   -1.0* (0 until x.length).map(i=>x(i)*log(z(i))+(1-x(i))*log(1-z(i))).reduce(_+_)
  }  
}


object LogisticRegression_breeze {
  def test_LR_simple() {
    var learning_rate: Double = 0.1
    val n_epochs: Int = 500
    val train_N: Int = 9
    val test_N: Int = 3
    val n_in: Int = 9
    val n_out: Int = 3
    val train_X_0: Array[Array[Double]] = Array(
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
    val train_X=matrix_utils.trans_array2breeze_2d(train_X_0)
    val train_Y_0: Array[Array[Double]] = Array(
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
    val train_Y=matrix_utils.trans_array2breeze_2d(train_Y_0)
    // construct
    val classifier = new LogisticRegression_breeze(n_in, n_out)
    // train
    var epoch: Int = 0
    var i: Int = 0
    var W_add:DenseMatrix[Double] =matrix_utils.build_2d_matrix((classifier.n_out, classifier.n_in))
    var b_add: DenseVector[Double] = matrix_utils.build_1d_matrix(classifier.n_out)
    for(epoch <- 0 until n_epochs) {
      for(i <- 0 until train_N) {
        val tmp=classifier.train(train_X(i,::).toDenseVector, train_Y(i,::).toDenseVector)
        W_add=W_add+tmp._1
        b_add=b_add+tmp._2
      }
      //updata w b
      W_add=W_add.map(x=>x/train_N)
      b_add=b_add.map(x=>x/train_N)
      //learning_rate *= 0.95
    }
    // test data
    val test_X_0: Array[Array[Double]] = Array(
      Array(1, 0, 1, 0, 0, 0,0,0,0),
      Array(0, 0, 1, 1, 1, 0,0,0,0),
      Array(0, 0, 0, 0, 0, 0,1,1,1)
    )
    val test_X=matrix_utils.trans_array2breeze_2d(test_X_0)
    val test_Y=matrix_utils.build_2d_matrix((test_N, n_out))
    // test
    var j: Int = 0
    for(i <- 0 until test_N) {
      var tmp=classifier.predict(test_X(i,::).toDenseVector)
      for(j <- 0 until n_out) {
        printf("%.5f ", tmp(j))
      }
      println()
      /*
       * 0.92048 0.04735 0.03217 
       * 0.01120 0.98404 0.00475 
       * 0.00196 0.00374 0.99430 
       * */
    }
  }
  
  def main(args: Array[String]) {
    test_LR_simple()
  }    
}