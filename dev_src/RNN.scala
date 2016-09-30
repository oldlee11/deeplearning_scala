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
  val log_layer = new LogisticRegression(hidden_layer_sizes(n_layers-1), n_out) 
  
  
  /**************** 
   * MLP的向前传播
   ****************/  
  //一个序列的第t时间状态下,经过mlp的向前传播
  def forward(x:Array[Double],
              dropout:Boolean=true, 
              p_dropout:Double=0.5){
    /*
     * step1 forward hidden_layers
     */
    var dropout_masks:ArrayBuffer[Array[Double]]=ArrayBuffer()
    var layer_inputs:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层hidden的输入
    var layer_doutputs:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层hidden的doutput
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
                                           pre_time_hidden_input=(if(i==0) null else layer_inputs(i-1)),
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
    /*
     * step2 forward lr
     */  
    
  }
  

  

  
}

object RNN {
  
}