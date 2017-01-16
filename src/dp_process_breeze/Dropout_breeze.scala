package dp_process_breeze

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
//import breeze.linalg._
//import breeze.numerics._

import scala.util.Random
import scala.math

import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

class Dropout_breeze(n_in_arg:Int,
                     hidden_layer_sizes_arg:Array[Int], 
                     n_out:Int,
                     rng_arg: Random=null, 
                     activation:String="ReLU") {
  /**********************
   ****   init       ****
   **********************/  
  val hidden_layer_sizes:Array[Int]=hidden_layer_sizes_arg
  val n_in:Int=n_in_arg
  val n_layers:Int=hidden_layer_sizes.length
  // var hidden_layer_sizes: Array[Int] = new Array[Int](n_layers)
  //n_layers个HiddenLayer  
  var hidden_layers: Array[HiddenLayer_breeze] = new Array[HiddenLayer_breeze](n_layers)
  val rng:Random =if(rng_arg == null) new Random(1234) else rng_arg
  // construct multi-layer
  //组合n层网络
  for(i <- 0 until n_layers) {
    var input_size: Int = 0
    if(i == 0) {
      input_size = n_in
    } else {
      input_size = hidden_layer_sizes(i-1)
    }
    // construct hidden_layer  ,注意这里以ReLu作为非线性映射函数,而不是sigmoid
    hidden_layers(i) = new HiddenLayer_breeze(input_size, hidden_layer_sizes(i), null, null, rng,activation)
    //print("hidden_layers_"+i+":"+hidden_layers(i).n_in+"\t"+hidden_layers(i).n_out+"\n")//debug
  }
  // layer for output using LogisticRegression
  //输出层用于监督学习使用LR
  val log_layer = new LogisticRegression_breeze(hidden_layer_sizes(n_layers-1), n_out)  
  //print("log_layer:"+log_layer.n_in+"\t"+log_layer.n_out+"\n")//debug  

  
  
  
  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)--------------没有优化
    x:一个样本的x数据,据取值为[0,1]之间
    y:样本的标准输出
             输出(Array[每一层的W_add_tmp,包括hidden和logistic层],
       Array[每一层的b_add_tmp,包括hidden和logistic层],
       cross_entropy值,
                       第一层hidden网络的d_v(如果是mlp则没有用处,如果是cnn则用于maxpool的反向梯度))
  */
  def train(x: DenseVector[Double],
            y:DenseVector[Double],
            dropout:Boolean=true, 
            p_dropout:Double=0.5):(Array[DenseMatrix[Double]],Array[DenseVector[Double]],Double,DenseVector[Double])={  
    val layers_train_W_add_tmp:ArrayBuffer[DenseMatrix[Double]]=ArrayBuffer()
    val layers_train_b_add_tmp:ArrayBuffer[DenseVector[Double]]=ArrayBuffer()  
    /*
     * step1 forward hidden_layers
     */
    var dropout_masks:ArrayBuffer[DenseVector[Double]]=ArrayBuffer()
    var layer_inputs:ArrayBuffer[DenseVector[Double]]=ArrayBuffer()
    var outputs_d_s:ArrayBuffer[DenseVector[Double]]=ArrayBuffer()
    var layer_input:DenseVector[Double]=DenseVector()
    for(i <-0 until n_layers){
      if(i==0) {
        layer_input=x 
      }
      layer_inputs +=layer_input
      //print("hidden_layer_"+i+"_forward\n")//debug
      val tmp=hidden_layers(i).forward(layer_input)//layer_input:本层的输入变为下一层的输出,注意layer_input会改变 ,但是layer_inputs内的元素是不变化的
      layer_input=tmp._1
      outputs_d_s+=tmp._2
      if(dropout){
         var mask:DenseVector[Double] = hidden_layers(i).dropout(size=layer_input.length, p=1-p_dropout)
         for(i <-0 until layer_input.length){
           layer_input(i) *= mask(i)  
         }  
         dropout_masks += mask
      }
    } 
    /*
     * step2 forward & backward log_layer
     */
    //print("log_layer train\n")//debug
    val log_layer_train=log_layer.train(x=layer_input,y=y)
    layers_train_W_add_tmp +=log_layer_train._1
    layers_train_b_add_tmp +=log_layer_train._2
    val log_layer_train_d_y:DenseVector[Double]=log_layer_train._3
    val train_cross_entropy_result:Double=log_layer_train._4
    /*
     * step3 backward hidden_layers
     */  
    var hidden_layers_i_train_d_v:DenseVector[Double]=DenseVector() 
    for(i <- (0 until n_layers).reverse){
      //print("hidden_layer_"+i+"_backward\n")//debug
      if (i == n_layers-1){
        //下一层hidden_layers(i+1)是LogisticRegression
        val hidden_layers_i_train=hidden_layers(i).backward(input=layer_inputs(i),
                                                            output_d_s=outputs_d_s(i),
                                                            next_layer_n_in=hidden_layer_sizes(n_layers-1),
                                                            next_layer_n_out=n_out,
                                                            next_layer_w=log_layer.W,
                                                            next_layer_dv=log_layer_train_d_y,
                                                            dropout=dropout,
                                                            mask=(if(dropout) dropout_masks(i) else DenseVector()));  
        layers_train_W_add_tmp +=hidden_layers_i_train._1
        layers_train_b_add_tmp +=hidden_layers_i_train._2
        hidden_layers_i_train_d_v=hidden_layers_i_train._3
      }else{
        //下一层hidden_layers(i+1)是HiddenLayer
        val hidden_layers_i_train=hidden_layers(i).backward(input=layer_inputs(i),
                                                            output_d_s=outputs_d_s(i),
                                                            next_layer_n_in= hidden_layer_sizes(i),
                                                            next_layer_n_out=hidden_layer_sizes(i+1),
                                                            next_layer_w=hidden_layers(i+1).W,
                                                            next_layer_dv=hidden_layers_i_train_d_v,
                                                            dropout=dropout,
                                                            mask=(if(dropout) dropout_masks(i) else DenseVector()))   
        layers_train_W_add_tmp +=hidden_layers_i_train._1
        layers_train_b_add_tmp +=hidden_layers_i_train._2
        hidden_layers_i_train_d_v=hidden_layers_i_train._3   
      }      
    }
    (layers_train_W_add_tmp.reverse.toArray,layers_train_b_add_tmp.reverse.toArray,train_cross_entropy_result,hidden_layers_i_train_d_v)
  }    
  
  /*训练,使用训练集做一个批次的BP训练
    lr:参数迭代的学习率
    inputs_x 训练集输入x
    inputs_y 训练集输出y
   * */  
  def train_batch(inputs_x: Array[DenseVector[Double]], 
                  inputs_y: Array[DenseVector[Double]],
                  lr: Double,
                  alpha:Double=0.9,
                  dropout:Boolean=true, 
                  p_dropout:Double=0.5,
                  batch_num_per:Double=1.0,
                  debug:Boolean=false)={
    /*
     * 抽样数据
     * */
    //抽取样本个数
    val batch_num:Int=if(batch_num_per==1.0){
      inputs_x.length
    }else{
      math.round((inputs_x.length*batch_num_per).toFloat)//每次批量训练样本数
    }
    val rng_epooch:Random=new Random()//每次生成一个种子
    val rng_index:ArrayBuffer[Int]=ArrayBuffer();
    if(batch_num_per==1.0){
      for(i <- 0 to (batch_num-1)) rng_index += i//抽样样本的角标 
    }else{
      for(i <- 0 to (batch_num-1)) rng_index += math.round((rng_epooch.nextDouble()*(inputs_x.length-1)).toFloat)//抽样样本的角标        
    }
    
    /*
     * mapper
     * 计算参数的add_tmp
     * */
    /*串行
     val train_batch_result=rng_index.map(i=>train(x=inputs_x(i),
                                                          y=inputs_y(i),
                                                          dropout=dropout, 
                                                          p_dropout=p_dropout))*/
    /*并行*/
    val train_batch_result=rng_index.par.map(i=>train(x=inputs_x(i),
                                                      y=inputs_y(i),
                                                      dropout=dropout, 
                                                      p_dropout=p_dropout))
                                                          
    /*
     * reduce---------没有优化
     * 更新参数
     * */
    //w
    val W_add_tmp_all:ArrayBuffer[DenseMatrix[Double]]=ArrayBuffer()
    var w_rows:Int=0;
    var w_cols:Int=0;
    for(i_layer<-0 until n_layers+1){//第i_layer层网络
      w_rows=train_batch_result(0)._1(i_layer).rows
      w_cols=train_batch_result(0)._1(i_layer).cols
      W_add_tmp_all+=matrix_utils.build_2d_matrix((w_rows,w_cols))
      for(w_row<-0 until w_rows){//第i_layer层网络中W的行数目
        for(w_col<-0 until w_cols){//第i_layer层网络中W的列数目
          for(i_train_time<-0 until train_batch_result.length){//第i_train_time个样本的训练结果
            W_add_tmp_all(i_layer)(w_row,w_col) += train_batch_result(i_train_time)._1(i_layer)(w_row,w_col)
          }
          W_add_tmp_all(i_layer)(w_row,w_col)=W_add_tmp_all(i_layer)(w_row,w_col)/batch_num//变化的平均值
          if(i_layer== n_layers){//logicit
            log_layer.W_add(w_row,w_col)=alpha*log_layer.W_add(w_row,w_col)+lr*W_add_tmp_all(i_layer)(w_row,w_col)
            log_layer.W(w_row,w_col) +=  log_layer.W_add(w_row,w_col)            
          }else{
            hidden_layers(i_layer).W_add(w_row,w_col)=alpha*hidden_layers(i_layer).W_add(w_row,w_col)+lr*W_add_tmp_all(i_layer)(w_row,w_col)
            hidden_layers(i_layer).W(w_row,w_col) +=  hidden_layers(i_layer).W_add(w_row,w_col)
          }
        }
      }
    }
    //b
    val b_add_tmp_all:ArrayBuffer[DenseVector[Double]]=ArrayBuffer()
    var b_rows:Int=0;    
    for(i_layer<-0 until n_layers+1){//第i_layer层网络
      b_rows=train_batch_result(0)._2(i_layer).length
      b_add_tmp_all+=matrix_utils.build_1d_matrix(b_rows)
      for(b_row<-0 until b_rows){//第i_layer层网络中b的行数目
        for(i_train_time<-0 until train_batch_result.length){//第i_train_time个样本的训练结果
          b_add_tmp_all(i_layer)(b_row) += train_batch_result(i_train_time)._2(i_layer)(b_row)
        }
        b_add_tmp_all(i_layer)(b_row)=b_add_tmp_all(i_layer)(b_row)/batch_num//变化的平均值
        if(i_layer== n_layers){//logicit
            log_layer.b_add(b_row)=alpha*log_layer.b_add(b_row)+lr*b_add_tmp_all(i_layer)(b_row)
            log_layer.b(b_row) +=  log_layer.b_add(b_row)            
        }else{
            hidden_layers(i_layer).b_add(b_row)=alpha*hidden_layers(i_layer).b_add(b_row)+lr*b_add_tmp_all(i_layer)(b_row)
            hidden_layers(i_layer).b(b_row) +=  hidden_layers(i_layer).b_add(b_row)
        }        
      }
    }
    //完成一批次训练后计算本批次的平均交叉嫡cross_entropy_result/batch_num (cross_entropy_result内部是累加的)
    println("cross_entropy="+train_batch_result.map(x=>x._3).reduce(_+_)/batch_num)   
  }
  
  
  //预测一个样本-------------没有优化
  //注意 没有返回值  ,直接改变y
  def predict(x: DenseVector[Double], dropout:Boolean=false,p_dropout:Double=0.5):DenseVector[Double]={
    var layer_input:DenseVector[Double] = x
    for(i <- 0 until n_layers){
      if(dropout){
        for(m <-0 until hidden_layers(i).n_out){
          for(n <-0 until hidden_layers(i).n_in){
            hidden_layers(i).W(m,n)=(1 - p_dropout)* hidden_layers(i).W(m,n)   
          }
        }
      }
      layer_input=hidden_layers(i).forward(layer_input)._1
    }
    log_layer.predict(layer_input)
  }  

  
  /* for time debug
   * */
  var fun_begin_time:Long=0
  def deal_begin_time()={
    fun_begin_time=System.currentTimeMillis()
  }
  def deal_end_time(fun_name:String)={
    print(fun_name+" use "+(System.currentTimeMillis()-fun_begin_time)+"毫秒\n")
    fun_begin_time=0
  }    
    
  
    
}

object Dropout_breeze {
  def test_Dropout_simple() {
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
    val train_X=train_X_0.map(x=>new DenseVector(x))//matrix_utils.trans_array2breeze_2d(train_X_0)
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
    val train_Y=train_Y_0.map(x=>new DenseVector(x))//matrix_utils.trans_array2breeze_2d(train_Y_0)
    val n_out:Int=3
    val classifier = new  Dropout_breeze(n_in_arg=9,hidden_layer_sizes_arg=Array(100,50), n_out=n_out,rng_arg=null,activation="ReLU")//实验成果了   hiddenlayer.scala 初始化 val a: Double =1/ math.pow(n_out,0.25)    learning_rate *=0.9   lr=0.1
    val n_epochs:Int=10
    val train_N:Int=train_Y.length
    var learning_rate:Double=0.1
    // train
    var epoch: Int = 0
    var i: Int = 0
    classifier.deal_begin_time()
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate,alpha=0.9, dropout=true, p_dropout=0.1, batch_num_per=1.0,debug=true)
      learning_rate *=0.9
    }
    classifier.deal_end_time("train_batch")
     // test data
    val test_X_0: Array[Array[Double]] = Array(
      Array(1, 1, 1, 0, 0, 0,0,0,0),
      Array(0, 0, 1, 1, 1, 0,0,0,0),
      Array(0, 0, 0, 0, 0, 0,1,1,1)
    )
    val test_X=test_X_0.map(x=>new DenseVector(x))
    val test_N:Int=test_X.length
    val test_Y: ArrayBuffer[DenseVector[Double]] =ArrayBuffer()  
    // test
    var j: Int = 0
    for(i <- 0 until test_N) {
      test_Y += classifier.predict(test_X(i))
      for(j <- 0 until n_out) {
        printf("%.5f ", test_Y(i)(j))
      }
      println()
    }   
/*
0.99995 0.00005 0.00000 
0.00000 1.00000 0.00000 
0.00000 0.00000 1.00000 
 * */    
  }  

  
  def main(args: Array[String]) {
    test_Dropout_simple()
  } 
}