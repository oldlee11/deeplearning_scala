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

/*
 * _n_in 输入层的长度
 * _hidden_layer_sizes 各个隐层层的个数
 * _n_out 输出层个数  
 * _win_times  时间框  即win_times
 * RNN_type RNN网络形态=full 表示每个时间状态的输入均对应一个输出,即每个时间状态的mlp均有输出
 *                  =one  表示一个输入序列(多个时间状态的输入)对应一个输出,即仅在最后时间状态的mlp才有输出
 * */
class RNN(_n_in:Int, 
              _hidden_layer_sizes:Array[Int], 
              _n_out:Int,
              _win_times:Int,
              RNN_type:String="full",
              _rng: Random=null,
              activation:String="sigmoid") {

  
  /**************** 
   * 全局变量
   ****************/  
  val rng=if(_rng==null) new Random(1234) else _rng
  val n_in:Int=_n_in
  val n_out:Int=_n_out
  val hidden_layer_sizes:Array[Int]=_hidden_layer_sizes
  val win_times:Int=_win_times
  val n_layers:Int=hidden_layer_sizes.length
  //缓存正向传播的结果数据(每个时间状态的都有缓存,所以是win_times个)
  val forward_results:Array[(Array[Double],Array[Double])]=new Array(win_times)
  //缓存反向传播的结果数据(每个时间状态的都有缓存,所以是win_times个)
  val backward_results:Array[(Array[Array[Double]],Array[Double],Array[Array[Double]],Array[Double],Array[Double])]=new Array(win_times)
  
  
  /**************** 
   * construct multi-layer
   * 组合n层网络
   ****************/
  var input_size: Int = 0
  var hidden_layers: Array[RNN_HiddenLayer] = new Array[RNN_HiddenLayer](n_layers)
  for(i <- 0 until n_layers) {
    if(i == 0) {
      input_size = n_in
    } else {
      input_size = hidden_layer_sizes(i-1)
    }
    // construct hidden_layer  ,注意这里以ReLu作为非线性映射函数,而不是sigmoid
    hidden_layers(i) = new RNN_HiddenLayer(_n_in=input_size,_n_out=hidden_layer_sizes(i),activation=activation)
    
  }
  // layer for output using LogisticRegression
  //输出层用于监督学习使用LR
  val log_layer = new RNN_LogisticRegression(hidden_layer_sizes(n_layers-1), n_out) 
  
  
  /**************** 
   * MLP的向前传播 
   * 一个序列的第t时间状态下,经过mlp的向前传播
   * 输入
   *   x 一个样本的输入x
   *   is_first_time            是否是第一个时间状态t0,如果是则pre_time_layers_x=null
   *   pre_time_layers_output   上一个时间状态中每一层hidden的输出数据
   *   is_forward_for_loglayer  是否要对最后的log层做向前传播?如果是则对log_layer做一次向前传播
   *   dropout 是否使用dropout
   *   p_dropout dropout的比例
   *   
   * 输出
   *   layer_inputs  每个一个层级的输入 以及最后一个log层的输出  共hidden_layers.length+2个数据
   *   layer_doutputs 每一个hidden层的doutput 共hidden_layers.length个数据
   *   dropout_masks  每一个hidden层的用于dropout的mask掩码
   *   
   *   
   *   例如下面一个结构:2层hidden + 1层log
   *   
   *        /\--------------layer_inputs[3]=log层的输出 
   *        ||
   *   logisticregression_layer(n_out个神经元) 
   *        /\--------------layer_inputs[2]=log层的输入=第1层hidden的输出
   *        ||
   *        ||--------------layer_doutputs[1]=第1层hidden的doutput=dactivation_fun(output)
   *   hidden_layer[1]
   *        /\--------------layer_inputs[1]=第1层hidden的输入=第0层hidden的输出
   *        ||
   *        ||--------------layer_doutputs[0]=第0层hidden的doutput=dactivation_fun(output)
   *   hidden_layer[0]
   *        /\--------------layer_inputs[0]=第0层hidden的输入=x
   *        ||
   *   input(n_in个神经元)
   ****************/  
  def forward_one_time(x:Array[Double],
                       is_first_time:Boolean,
                       pre_time_layers_output:Array[Array[Double]]=null,
                       is_forward_for_loglayer:Boolean,
                       dropout:Boolean=true, 
                       p_dropout:Double=0.3):(Array[Array[Double]],Array[Array[Double]],Array[Array[Double]])={
    /*
     * step1 forward hidden_layers
     */
    var dropout_masks:ArrayBuffer[Array[Double]]=ArrayBuffer()
    var layer_inputs:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层hidden的输入
    var layer_doutputs:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层hidden的doutput输出
    var layer_input:Array[Double]=Array()
    var layer_doutput:Array[Double]=Array()
    var hidden_forward_result:(Array[Double],Array[Double])=(Array(),Array())
    for(i <-0 until n_layers){
      if(i==0) {
        layer_input=x 
      }
      layer_inputs +=layer_input
      //对第i层hidden_layers做向前传播
      hidden_forward_result=hidden_layers(i).forward(input=layer_input,
                                           is_first_time=is_first_time,
                                           pre_time_hidden_input=(if(is_first_time) null else pre_time_layers_output(i)),
                                           W_arg=hidden_layers(i).W,
                                           b_arg=hidden_layers(i).b,
                                           W_hh_arg=hidden_layers(i).W_hh,
                                           b_hh_arg=hidden_layers(i).b_hh)
      layer_input=hidden_forward_result._1
      layer_doutput=hidden_forward_result._2
      layer_doutputs +=layer_doutput
      if(dropout){
         var mask:Array[Double] = hidden_layers(i).dropout(size=layer_input.length, p=1-p_dropout)
         for(i <-0 until layer_input.length){
           layer_input(i) *= mask(i)  
         }  
         dropout_masks += mask
      }
    }
    layer_inputs +=layer_input
    /*
     * step2 forward log_layer
     */  
    if(is_forward_for_loglayer==true){
      layer_inputs+=log_layer.forward(x=layer_input)
    }
    /*
     * 输出
     * */
    (layer_inputs.toArray,layer_doutputs.toArray,dropout_masks.toArray)
  }

  /**************** 
   * MLP的向后传播 
   * 一个序列的第t时间状态下,经过mlp的向后传播
   * 输入
   *   y               一个样本的y
   *   layer_inputs    一个样本x在时间状态t下经过向前传播(forward_one_time函数)输出的每个层的输入以及log的输出,详细见【forward_one_time】函数
   *   layer_doutputs  一个样本x在时间状态t下经过向前传播(forward_one_time函数)输出的每个hidden层的输出doutput,详细见【forward_one_time】函数
   *   dropout_masks   一个样本x在时间状态t下经过向前传播(forward_one_time函数)输出的每个hidden层的用于dropout的掩码,详细见【forward_one_time】函数
   *   is_first_time   是否是第一个时间状态t0,如果是则pre_time_hidden_inputs=null
   *   pre_time_layers_output 是前一个时间的各个hidden的输出
   *   is_last_time    是否是最后一个时间状态
   *   next_time_dv              同一层的上一时刻的局部梯度dv
   *   is_backward_for_loglayer   是否对loglayer做向后传播
   * 
   *   例如下面一个结构:2层hidden + 1层log
   *   
   *        ||
   *        \/
   *   logisticregression_layer(n_out个神经元) 
   *        ||----------------------------------经过log_layer层反向传播后的局部梯度变化(一个样本的更新):layers_train_W_add_tmp[2],layers_train_b_add_tmp[2]
   *        ||
   *        \/
   *   hidden_layer[1]
   *        ||----------------------------------经过hidden1层反向传播后的局部梯度变化(一个样本的更新):layers_train_W_add_tmp[1],layers_train_b_add_tmp[1],,layers_train_W_hh_add_tmp[1],layers_train_b_hh_add_tmp[1]
   *        ||
   *        \/
   *   hidden_layer[0]
   *        ||----------------------------------经过hidden0层反向传播后的局部梯度变化(一个样本的更新):layers_train_W_add_tmp[0],layers_train_b_add_tmp[0],layers_train_W_hh_add_tmp[0],layers_train_b_hh_add_tmp[0]
   *        ||
   *        \/
   *   input(n_in个神经元)
   ****************/
  def backward_one_time(y:Array[Int],
                        layer_inputs:Array[Array[Double]],
                        layer_doutputs:Array[Array[Double]],
                        dropout_masks:Array[Array[Double]],
                        is_first_time:Boolean,
                        pre_time_layers_output:Array[Array[Double]]=null,
                        is_last_time:Boolean,
                        next_time_dv:Array[Array[Double]],
                        is_backward_for_loglayer:Boolean,
                        dropout:Boolean=true):(Array[Array[Array[Double]]],Array[Array[Double]],Array[Array[Array[Double]]],Array[Array[Double]])={
    /*
     * 初始化输出数据
     * */
    val layers_train_W_add_tmp:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()//每一层的局部梯度w_add_tmp
    val layers_train_b_add_tmp:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层的局部梯度b_add_tmp 
    val layers_train_W_hh_add_tmp:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()//每一层的局部梯度W_hh_add_tmp
    val layers_train_b_hh_add_tmp:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层的局部梯度h_hh_add_tmp
    
    /*
     * step1 backward log_layer
     */
    var log_backward_result:(Array[Array[Double]],Array[Double],Array[Double],Double)=null
    var train_cross_entropy_result:Double=0.0
    if(is_backward_for_loglayer==false){
      log_backward_result=log_layer.backward(p_y_given_x_softmax=layer_inputs(n_layers+1), 
                                             x=layer_inputs(n_layers),
                                             y=y)
      layers_train_W_add_tmp += log_backward_result._1
      layers_train_b_add_tmp += log_backward_result._2
      train_cross_entropy_result=log_backward_result._4
    }
    
    /*
     * step2 backward hidden_layers
     */  
    var hidden_layers_i_train_d_v:Array[Double]=Array() 
    for(i <-(0 until n_layers).reverse){
      val hidden_layers_i_train=
        if (i == n_layers-1){
          //下一层hidden_layers(i+1)是LogisticRegression
          if(is_backward_for_loglayer==true){
            //loglayer输出层做了向后传播
            hidden_layers(i).backward(input=layer_inputs(i), 
                                      doutput=layer_doutputs(i),
                                      is_first_time=is_first_time,
                                      pre_time_hidden_input=if(is_first_time) null else pre_time_layers_output(i),
                                      is_last_time=is_last_time, 
                                      next_time_w=if(is_last_time) null else hidden_layers(i).W_hh,
                                      next_time_dv=if(is_last_time) null else next_time_dv(i),
                                      is_last_layer=false, 
                                      next_layer_w=log_layer.W, 
                                      next_layer_dv=log_backward_result._3, 
                                      dropout=dropout, 
                                      mask=if(dropout) dropout_masks(i) else Array())   
          }else{
            //loglayer输出层没有做向后传播
            hidden_layers(i).backward(input=layer_inputs(i), 
                                      doutput=layer_doutputs(i),
                                      is_first_time=is_first_time,
                                      pre_time_hidden_input=if(is_first_time) null else pre_time_layers_output(i),
                                      is_last_time=is_last_time, 
                                      next_time_w=if(is_last_time) null else hidden_layers(i).W_hh,
                                      next_time_dv=if(is_last_time) null else next_time_dv(i),
                                      is_last_layer=true, 
                                      next_layer_w=null, 
                                      next_layer_dv=null, 
                                      dropout=dropout, 
                                      mask=if(dropout) dropout_masks(i) else Array())            
          }                                                      
        }else{
          //下一层hidden_layers(i+1)是hidden
          hidden_layers(i).backward(input=layer_inputs(i), 
                                    doutput=layer_doutputs(i),
                                    is_first_time=is_first_time,
                                    pre_time_hidden_input=if(is_first_time) null else pre_time_layers_output(i),
                                    is_last_time=is_last_time, 
                                    next_time_w=if(is_last_time) null else hidden_layers(i).W_hh,
                                    next_time_dv=if(is_last_time) null else next_time_dv(i),
                                    is_last_layer=false, 
                                    next_layer_w=hidden_layers(i+1).W, 
                                    next_layer_dv=hidden_layers_i_train_d_v, 
                                    dropout=dropout, 
                                    mask=if(dropout) dropout_masks(i) else Array())        
        }
        layers_train_W_add_tmp +=hidden_layers_i_train._1
        layers_train_b_add_tmp +=hidden_layers_i_train._2
        layers_train_W_hh_add_tmp +=hidden_layers_i_train._3
        layers_train_b_hh_add_tmp +=hidden_layers_i_train._4
        hidden_layers_i_train_d_v=hidden_layers_i_train._5      
    }
    (layers_train_W_add_tmp.reverse.toArray,layers_train_b_add_tmp.reverse.toArray,layers_train_W_hh_add_tmp.reverse.toArray,layers_train_b_hh_add_tmp.reverse.toArray)
  }
  

  
}

object RNN {
  
}