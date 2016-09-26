package dp_process_parallel

import scala.util.Random
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array
import scala.math

import java.io.File; 
import java.io._
import scala.io.Source

  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

/*
 * input_size           处理图片的长和高
 * n_channel            图片组成的个数,一般一个图片是以RGB形式给出的,所以一般=3
 * output_size          分类个数
 * n_kernel_Array       每一个 ConvPoolLayer层内卷积层的n_kernel
 * kernel_size_Array    每一个 ConvPoolLayer层内卷积层的kernel_size
 * pool_size_Array      每一个 ConvPoolLayer层内池化层的pool_size
 * activation           每一个 ConvPoolLayer层内卷积层的非线性函数
 * */
class CNN_parallel(input_size:(Int,Int),
          output_size:Int,
          n_kernel_Array:Array[Int],
          kernel_size_Array:Array[(Int,Int)],
          pool_size_Array:Array[(Int,Int)],
          n_hidden:Int,
          n_channel:Int=3,
          _rng: Random=null,
          activation:String="ReLU",
          activation_mlp:String="tanh"){

  var rng:Random=if(_rng == null) new Random(1234) else _rng
    
  //ConvPoolLayer层个数
  val n_ConvPoolLayer=n_kernel_Array.length
  val ConvPoolLayer_layers:Array[ConvPoolLayer_parallel]= new Array[ConvPoolLayer_parallel](n_ConvPoolLayer)
  //建立ConvPoolLayer_layers
  for(i <-0 until n_ConvPoolLayer){
    if(i==0){  
      ConvPoolLayer_layers(i)=new ConvPoolLayer_parallel(input_size_in=input_size,
                                                n_kernel_in=n_kernel_Array(i),
                                                kernel_size_in=kernel_size_Array(i),
                                                pool_size_in=pool_size_Array(i),
                                                n_channel_in=n_channel,
                                                _rng=rng,
                                                activation=activation)  
    }else{
      ConvPoolLayer_layers(i)=new ConvPoolLayer_parallel(input_size_in=(ConvPoolLayer_layers(i-1).pool_s0,ConvPoolLayer_layers(i-1).pool_s1),
                                                n_kernel_in=n_kernel_Array(i),
                                                kernel_size_in=kernel_size_Array(i),
                                                pool_size_in=pool_size_Array(i),
                                                n_channel_in=ConvPoolLayer_layers(i-1).n_kernel,
                                                _rng=rng,
                                                activation=activation)        
    }
  }
  //建立输出层  一成hidden+一层logistic
  val mlp_layer = new Dropout_parallel(n_in_arg=ConvPoolLayer_layers(n_ConvPoolLayer-1).n_kernel*ConvPoolLayer_layers(n_ConvPoolLayer-1).pool_s0*ConvPoolLayer_layers(n_ConvPoolLayer-1).pool_s1, 
                                       hidden_layer_sizes_arg=Array(n_hidden),
                                       n_out=output_size,
                                       activation=activation_mlp)  
  
  
  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)
    x:一个样本的x数据,据取值为[0,1]之间
    y:样本的标准输出
             输出((Array[convpool  每一层的W_add_tmp],Array[convpool  每一层的b_add_tmp]),
       (Array[mlp  每一层的W_add_tmp(包括hidden和logistic层)],Array[mlp  每一层的b_add_tmp(包括hidden和logistic层)])
       cross_entropy值)
  */
  def train(x:Array[Array[Array[Double]]],
            y:Array[Int],
            dropout:Boolean=true, 
            p_dropout:Double=0.5,
            debug_time:Boolean=false):((Array[Array[Array[Array[Array[Double]]]]],Array[Array[Double]]),(Array[Array[Array[Double]]],Array[Array[Double]]),Double)={  

    /*
     * step1 convpool layers forward 
     */
    var convolved_inputs:ArrayBuffer[Array[Array[Array[Double]]]]=ArrayBuffer()
    var layer_inputs:ArrayBuffer[Array[Array[Array[Double]]]]=ArrayBuffer()
    var max_index_xs:ArrayBuffer[Array[Array[Array[(Int,Int)]]]]=ArrayBuffer()
    var layer_input:Array[Array[Array[Double]]]=Array()    
    for(i <-0 until n_ConvPoolLayer){
      if(i==0) { 
        layer_input=x       
      }
      layer_inputs +=layer_input
      val convpool_layers_i_forward=ConvPoolLayer_layers(i).cnn_forward(layer_input,debug_time=debug_time)
      max_index_xs +=convpool_layers_i_forward._3
      layer_input=convpool_layers_i_forward._2//pooled_input
      convolved_inputs +=convpool_layers_i_forward._1
    }
    /*
     * step2 mlp train
     */
    val layer_input_copy_one:Array[Double]=flatten(layer_input)//打平
    val mlp_layer_train=mlp_layer.train(x=layer_input_copy_one,
                                        y=y,
                                        dropout=dropout, 
                                        p_dropout=p_dropout)
    val mlp_layers_train_W_add_tmp = mlp_layer_train._1
    val mlp_layers_train_b_add_tmp = mlp_layer_train._2
    val train_cross_entropy_result:Double=mlp_layer_train._3    
    val mlp_firstlayer_train_d_v:Array[Double]=mlp_layer_train._4//mlp层的第一个hidden层的d_v,用于后续convpool的bp

    /*
     * step3 backward ConvPoolLayer_layers
     */   
    val convpool_layers_train_W_add_tmp:ArrayBuffer[Array[Array[Array[Array[Double]]]]]=ArrayBuffer()
    val convpool_layers_train_b_add_tmp:ArrayBuffer[Array[Double]]=ArrayBuffer()   
    var convpool_layers_i_train_d_v:Array[Array[Array[Double]]]=Array() 
    for(i <- (0 until n_ConvPoolLayer).reverse){
      if (i == n_ConvPoolLayer-1){
        //下一层ConvPoolLayer_layers(i+1)是Hidden
        val convpool_layers_i_train=ConvPoolLayer_layers(i).cnn_backward(x=layer_inputs(i),
                                             next_layer_type="hidden",
                                             next_layer_hidden_n_in=mlp_layer.n_in,
                                             next_layer_hidden_n_out=mlp_layer.hidden_layer_sizes(0),
                                             next_layer_hidden_w=mlp_layer.hidden_layers(0).W,                   
                                             next_layer_hidden_dv=mlp_firstlayer_train_d_v,
                                             poollayer_max_index_x=max_index_xs(i),
                                             convlayer_convolved_input=convolved_inputs(i))
        convpool_layers_train_W_add_tmp += convpool_layers_i_train._1
        convpool_layers_train_b_add_tmp += convpool_layers_i_train._2
        convpool_layers_i_train_d_v=convpool_layers_i_train._3
      } else{
        //下一层ConvPoolLayer_layers(i+1)是ConvPoolLayer_layers
        val convpool_layers_i_train=ConvPoolLayer_layers(i).cnn_backward(x=layer_inputs(i),
                                             next_layer_type="ConvLayer",
                                             next_layer_convlayer_n_channel=ConvPoolLayer_layers(i+1).n_channel,
                                             next_layer_convlayer_n_kernel=ConvPoolLayer_layers(i+1).n_kernel,
                                             next_layer_convlayer_kernel_size=ConvPoolLayer_layers(i+1).kernel_size,
                                             next_layer_convlayer_w=ConvPoolLayer_layers(i+1).W,
                                             next_layer_convlayer_dv=convpool_layers_i_train_d_v,
                                             poollayer_max_index_x=max_index_xs(i),
                                             convlayer_convolved_input=convolved_inputs(i))
        convpool_layers_train_W_add_tmp += convpool_layers_i_train._1
        convpool_layers_train_b_add_tmp += convpool_layers_i_train._2
        convpool_layers_i_train_d_v=convpool_layers_i_train._3         
      }  
    }    
    ((convpool_layers_train_W_add_tmp.reverse.toArray,convpool_layers_train_b_add_tmp.reverse.toArray),(mlp_layers_train_W_add_tmp,mlp_layers_train_b_add_tmp),train_cross_entropy_result)
  }    
  
  
  
  def train_batch(inputs_x: Array[Array[Array[Array[Double]]]],
                  inputs_y: Array[Array[Int]],
                  lr: Double,
                  batch_num_per:Double=1.0,
                  alpha:Double=0.0,
                  dropout:Boolean=true, 
                  p_dropout:Double=0.5,                  
                  debug:Boolean=false,
                  debug_time:Boolean=false)={
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
     * */
    /*串行
    val train_batch_result=rng_index.map(i=>train(x=inputs_x(i),
                                                  y=inputs_y(i),
                                                  dropout=dropout, 
                                                  p_dropout=p_dropout,
                                                  debug_time=debug_time))*/
    /*并行*/
    val train_batch_result=rng_index.par.map(i=>train(x=inputs_x(i),
                                                  y=inputs_y(i),
                                                  dropout=dropout, 
                                                  p_dropout=p_dropout,
                                                  debug_time=debug_time))
                                                  
                                                  
    /*reducer-mlp*/
    val mlp_train_W_add_tmp_all:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()
    val mlp_train_b_add_tmp_all:ArrayBuffer[Array[Double]]=ArrayBuffer()
    var b_rows:Int=0;       
    var w_rows:Int=0;
    var w_cols:Int=0;
    //mlp w    
    for(i_layer<-0 until mlp_layer.n_layers+1){//第i_layer层网络
      w_rows=train_batch_result(0)._2._1(i_layer).length
      w_cols=train_batch_result(0)._2._1(i_layer)(0).length
      mlp_train_W_add_tmp_all+=Array.ofDim[Double](w_rows,w_cols)
      for(w_row<-0 until w_rows){//第i_layer层网络中W的行数目
        for(w_col<-0 until w_cols){//第i_layer层网络中W的列数目
          for(i_train_time<-0 until train_batch_result.length){//第i_train_time个样本的训练结果
            mlp_train_W_add_tmp_all(i_layer)(w_row)(w_col) += train_batch_result(i_train_time)._2._1(i_layer)(w_row)(w_col)
          }
          mlp_train_W_add_tmp_all(i_layer)(w_row)(w_col)=mlp_train_W_add_tmp_all(i_layer)(w_row)(w_col)/batch_num//变化的平均值
          if(i_layer== mlp_layer.n_layers){//logicit
            mlp_layer.log_layer.W_add(w_row)(w_col)=alpha*mlp_layer.log_layer.W_add(w_row)(w_col)+lr*mlp_train_W_add_tmp_all(i_layer)(w_row)(w_col)
            mlp_layer.log_layer.W(w_row)(w_col) +=  mlp_layer.log_layer.W_add(w_row)(w_col)            
          }else{
            mlp_layer.hidden_layers(i_layer).W_add(w_row)(w_col)=alpha*mlp_layer.hidden_layers(i_layer).W_add(w_row)(w_col)+lr*mlp_train_W_add_tmp_all(i_layer)(w_row)(w_col)
            mlp_layer.hidden_layers(i_layer).W(w_row)(w_col) +=  mlp_layer.hidden_layers(i_layer).W_add(w_row)(w_col)
          }
        }
      }
    } 
    //mlp b 
    for(i_layer<-0 until mlp_layer.n_layers+1){//第i_layer层网络
      b_rows=train_batch_result(0)._2._2(i_layer).length
      mlp_train_b_add_tmp_all+=new Array[Double](b_rows)
      for(b_row<-0 until b_rows){//第i_layer层网络中b的行数目
        for(i_train_time<-0 until train_batch_result.length){//第i_train_time个样本的训练结果
          mlp_train_b_add_tmp_all(i_layer)(b_row) += train_batch_result(i_train_time)._2._2(i_layer)(b_row)
        }
        mlp_train_b_add_tmp_all(i_layer)(b_row)=mlp_train_b_add_tmp_all(i_layer)(b_row)/batch_num//变化的平均值
        if(i_layer== mlp_layer.n_layers){//logicit
            mlp_layer.log_layer.b_add(b_row)=alpha*mlp_layer.log_layer.b_add(b_row)+lr*mlp_train_b_add_tmp_all(i_layer)(b_row)
            mlp_layer.log_layer.b(b_row) +=  mlp_layer.log_layer.b_add(b_row)            
        }else{
            mlp_layer.hidden_layers(i_layer).b_add(b_row)=alpha*mlp_layer.hidden_layers(i_layer).b_add(b_row)+lr*mlp_train_b_add_tmp_all(i_layer)(b_row)
            mlp_layer.hidden_layers(i_layer).b(b_row) +=  mlp_layer.hidden_layers(i_layer).b_add(b_row)
        }        
      }
    } 
    
    /*reducer-convpool*/
    val convpool_train_W_add_tmp_all:ArrayBuffer[Array[Array[Array[Array[Double]]]]]=ArrayBuffer()
    val convpool_train_b_add_tmp_all:ArrayBuffer[Array[Double]]=ArrayBuffer()    
    b_rows=0;
    var w_kernels:Int=0;
    var w_channels:Int=0;
    w_rows=0;
    w_cols=0;
    //convpool W
    for(i_layer<-0 until n_ConvPoolLayer){//第i_layer层网络
      w_kernels=train_batch_result(0)._1._1(i_layer).length
      w_channels=train_batch_result(0)._1._1(i_layer)(0).length
      w_rows=train_batch_result(0)._1._1(i_layer)(0)(0).length
      w_cols=train_batch_result(0)._1._1(i_layer)(0)(0)(0).length
      convpool_train_W_add_tmp_all+=Array.ofDim[Double](w_kernels,w_channels,w_rows,w_cols)
      for(w_kernel<-0 until w_kernels){
        for(w_channel<-0 until w_channels){
          for(w_row<-0 until w_rows){//第i_layer层网络中W的行数目
            for(w_col<-0 until w_cols){//第i_layer层网络中W的列数目
              for(i_train_time<-0 until train_batch_result.length){//第i_train_time个样本的训练结果
                convpool_train_W_add_tmp_all(i_layer)(w_kernel)(w_channel)(w_row)(w_col) += train_batch_result(i_train_time)._1._1(i_layer)(w_kernel)(w_channel)(w_row)(w_col)
              }
              convpool_train_W_add_tmp_all(i_layer)(w_kernel)(w_channel)(w_row)(w_col)=convpool_train_W_add_tmp_all(i_layer)(w_kernel)(w_channel)(w_row)(w_col)/batch_num//变化的平均值
              ConvPoolLayer_layers(i_layer).W_add(w_kernel)(w_channel)(w_row)(w_col)=alpha*ConvPoolLayer_layers(i_layer).W_add(w_kernel)(w_channel)(w_row)(w_col)+lr*convpool_train_W_add_tmp_all(i_layer)(w_kernel)(w_channel)(w_row)(w_col)
              ConvPoolLayer_layers(i_layer).W(w_kernel)(w_channel)(w_row)(w_col) +=  ConvPoolLayer_layers(i_layer).W_add(w_kernel)(w_channel)(w_row)(w_col)
            }
          }
        }
      }
    }  
    //convpool b
    for(i_layer<-0 until n_ConvPoolLayer){//第i_layer层网络
      b_rows=train_batch_result(0)._1._2(i_layer).length
      convpool_train_b_add_tmp_all+=Array.ofDim[Double](w_kernels)
      for(b_row<-0 until b_rows){
        for(i_train_time<-0 until train_batch_result.length){//第i_train_time个样本的训练结果
          convpool_train_b_add_tmp_all(i_layer)(b_row) += train_batch_result(i_train_time)._1._2(i_layer)(b_row)
        }
        convpool_train_b_add_tmp_all(i_layer)(b_row)=convpool_train_b_add_tmp_all(i_layer)(b_row)/batch_num//变化的平均值
        ConvPoolLayer_layers(i_layer).b_add(b_row)=alpha*ConvPoolLayer_layers(i_layer).b_add(b_row)+lr*convpool_train_b_add_tmp_all(i_layer)(b_row)
        ConvPoolLayer_layers(i_layer).b(b_row) +=  ConvPoolLayer_layers(i_layer).b_add(b_row)
      }   
    }   
    //完成一批次训练后计算本批次的平均交叉嫡cross_entropy_result/batch_num (cross_entropy_result内部是累加的)
    println("cross_entropy="+train_batch_result.map(x=>x._3).reduce(_+_)/batch_num)      
  } 
    
    /*
     * 函数
     * */
  //把convpool层输出打平 移交给输出层
  def flatten(layer_input_in:Array[Array[Array[Double]]]):Array[Double]={
    val layer_input_dim0:Int=layer_input_in.length//输出层的上一次 convpoollayer层的n_kernel 
    val layer_input_dim1:Int=layer_input_in(0).length//输出层的上一次 convpoollayer层的输出的高 =s0
    val layer_input_dim2:Int=layer_input_in(0)(0).length//输出层的上一次 convpoollayer层的输出的长 =s1
    var index:Int=0
    //print(layer_input_dim0+"\t"+layer_input_dim1+"\t"+layer_input_dim2+"\n")//debug
    var layer_input_copy_one_out:Array[Double]=new Array(layer_input_dim0*layer_input_dim1*layer_input_dim2)
    //把3维的layer_input转化为1维的layer_input_copy_one
    for (k <- 0 until layer_input_dim0){
      for(i <- 0 until layer_input_dim1){
        for(j <-0 until layer_input_dim2){
          layer_input_copy_one_out(index)=layer_input_in(k)(i)(j)  
          index +=1
        }
      }
    }   
    layer_input_copy_one_out   
  }    

  //预测一个样本
  //注意 没有返回值  ,直接改变y
  def predict(x: Array[Array[Array[Double]]]):Array[Double]={
    var layer_input:Array[Array[Array[Double]]] = x
    for(i <- 0 until n_ConvPoolLayer){
      layer_input=ConvPoolLayer_layers(i).cnn_forward(layer_input)._2
    }
    
    val layer_input_copy_one=flatten(layer_input)//打平  
    mlp_layer.predict(layer_input_copy_one)
  }  
  
}


object CNN_parallel {
  
  def test_CNN_simple() {
    val train_X: Array[Array[Array[Array[Double]]]] = Array(
    Array(
      Array(
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)
      ),
      Array(
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0), 
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)
      )    
    ),
    Array(
      Array(
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0)  ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0) 
      ),
      Array(
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0), 
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)
      )     
    ),    
    Array(
      Array(
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0)  ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0) 
      ),
      Array(
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0), 
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)
      )     
    ),
    Array(
      Array(
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1)  ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0) 
      ),
      Array(
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0), 
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)
      )   
    ),    
    Array(
      Array(
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)   
      ),
      Array(
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0), 
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)
      )    
    ), 
    Array(
      Array(
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)   
      ),
      Array(
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0), 
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)
      )   
    ), 
    Array(
      Array(
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)  
      ),          
      Array(
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1)  ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)  
      )  
    ),   
    Array(
      Array(
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0)  ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0) 
      ),          
      Array(
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)   
      ) 
    )  ,   
    Array(
      Array(
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)  
      ),          
      Array(
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1)  ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)  
      )   
    )   
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
    val n_out:Int=3   
    val classifier = new  CNN_parallel(input_size=(9,9),output_size=n_out,n_kernel_Array=Array(15,20),kernel_size_Array=Array((2,2),(3,3)),pool_size_Array=Array((2,2),(2,2)),n_channel=2,n_hidden=20,_rng=null,activation="ReLU",activation_mlp="tanh")//n_epochs=250 alpha=0.0 learning_rate *=0.99  lr=0.1
                                                                                                                                                                                                                                              //hidden val a: Double = 4 * math.sqrt(6.0/(n_in + n_out))
                                                                                                                                                                                                                                              //cnn    val init_a_tmp:Double=1/ math.pow(f_out_tmp,0.25)
    val n_epochs:Int=250
    val train_N:Int=train_Y.length
    var learning_rate:Double=0.1
    // train
    var epoch: Int = 0
    var i: Int = 0
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate, batch_num_per=1.0, alpha=0.0)
      learning_rate *=0.99
    }
    
     // test data
    val test_X: Array[Array[Array[Array[Double]]]] = Array(
    Array(
      Array(
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)
      ),
      Array(
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0), 
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)        
      ) 
    ),
    Array(
      Array(
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1, 1, 1, 0, 0, 0,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)        
      ),          
      Array(
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1) ,
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)        
      ) 
    ) ,
    Array(
      Array(
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(0, 0, 0, 0, 0, 0,1,1,1),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)        
      ),
      Array(
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0),
        Array(0, 0, 0, 1, 1, 1,0, 0, 0), 
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0),
        Array(0, 0, 0, 1, 1, 1,0,0,0),
         Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0),
        Array(1,1,1,1,1,1,0,0,0)       
      )   
    )    
    )
    val test_N:Int=test_X.length
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
   * 应该输出
   * Array(1, 0,0),
   * Array(0, 0,1),
   * Array(0, 1,0)
   * 最后输出(2层)
0.98459 0.00537 0.01004 
0.01091 0.00449 0.98459 
0.00477 0.99152 0.00370 
   * */      
    }    
  }  
  
def train_test_mnist() {
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
    val n_epochs: Int = 400
    
    //lenet5
    val classifier = new  CNN_parallel(input_size=(height,width),output_size=10,n_kernel_Array=Array(6,16,120),kernel_size_Array=Array((5,5),(5,5),(4,4)),pool_size_Array=Array((2,2),(2,2),(1,1)),n_channel=1,n_hidden=84,_rng=null,activation="ReLU",activation_mlp="tanh")//200次迭代  lr=0.1  lr不变      91.2%
                                                                                                                                                                                                                                                                             //400次迭代 96.08%
    
    // train
    var epoch: Int = 0
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate, batch_num_per=0.01,alpha=0.0)
      //learning_rate *=0.99
    } 
    
    /*
     * test
     * */
    val filePath_test:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt"  
    val test_X:Array[Array[Array[Array[Double]]]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>{val tmp:Array[Array[Double]]=Array.ofDim[Double](height,width);for(i <- 0 until height){for(j <-0 until width){tmp(i)(j)=x._2(i*width+j)}};Array(tmp)})
    val test_N: Int = test_X.length
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
    //test_CNN_simple()
    train_test_mnist()//ok
  }   
}    