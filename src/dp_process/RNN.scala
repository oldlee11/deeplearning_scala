package dp_process

import scala.util.Random
import scala.math
import scala.math.round

import scala.collection.mutable.ArrayBuffer    //用于建立可变的array


  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

/* 
 * 大致示意图-1(每个时刻的mlp均有输出     _RNN_structure='full')
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
 * 大致示意图-2(只在最后时刻的mlp有输出   _RNN_structure='one')
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
 * _RNN_structure RNN网络形态=full 表示每个时间状态的输入均对应一个输出,即每个时间状态的mlp均有输出
 *                        =one  表示一个输入序列(多个时间状态的输入)对应一个输出,即仅在最后时间状态的mlp才有输出
 * */
class RNN(_n_in:Int, 
          _hidden_layer_sizes:Array[Int], 
          _n_out:Int,
          _win_times:Int,
          _RNN_structure:String="one",
          _rng: Random=null,
          activation:String="ReLU") {

  
  /**************** 
   * 全局变量
   ****************/  
  val rng=if(_rng==null) new Random(1234) else _rng
  val n_in:Int=_n_in
  val n_out:Int=_n_out
  val hidden_layer_sizes:Array[Int]=_hidden_layer_sizes
  val win_times:Int=_win_times
  val n_layers:Int=hidden_layer_sizes.length
  val RNN_structure:String=_RNN_structure
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
                       p_dropout:Double=0.3,
                       debug:Boolean=false):(Array[Array[Double]],Array[Array[Double]],Array[Array[Double]])={
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
      if(debug) {print("layer_input=");layer_input.map(x=>print(x+"\t"));print("\n")}
      if(debug) print("hidden "+i+"\n")
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
    if(debug) {print("layer_input=");layer_input.map(x=>print(x+"\t"));print("\n")}
    /*
     * step2 forward log_layer
     */  
    if(is_forward_for_loglayer==true){
      if(debug) print("log_layer \n")
      layer_inputs+=log_layer.forward(x=layer_input)
      if(debug) {print("layer_input=");layer_inputs(hidden_layers.length+1).map(x=>print(x+"\t"));print("\n")}
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
   *   is_backward_for_loglayer   是否对loglayer做向后传播 如果=false 即无log反向传播 则y可以为null
   * 
   * 输出
   *   layers_train_W_add_tmp     每个hidden和log,在训练了一个样本后的w增量
   *   layers_train_b_add_tmp     每个hidden和log,在训练了一个样本后的b增量
   *   layers_train_W_hh_add_tmp  每个hidden,在训练了一个样本后的w_hh增量
   *   layers_train_b_hh_add_tmp  每个hidden,在训练了一个样本后的b_hh增量
   *   layers_train_dv            每个hidden,在训练了一个样本后的局部梯度
   *   train_cross_entropy_result
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
   *        ||                                  layers_train_dv[1]
   *        \/
   *   hidden_layer[0]
   *        ||----------------------------------经过hidden0层反向传播后的局部梯度变化(一个样本的更新):layers_train_W_add_tmp[0],layers_train_b_add_tmp[0],layers_train_W_hh_add_tmp[0],layers_train_b_hh_add_tmp[0]
   *        ||                                  layers_train_dv[0]
   *        \/
   *   input(n_in个神经元)
   ****************/
  def backward_one_time(y:Array[Int]=null,
                        layer_inputs:Array[Array[Double]],
                        layer_doutputs:Array[Array[Double]],
                        dropout_masks:Array[Array[Double]],
                        is_first_time:Boolean,
                        pre_time_layers_output:Array[Array[Double]]=null,
                        is_last_time:Boolean,
                        next_time_dv:Array[Array[Double]],
                        is_backward_for_loglayer:Boolean,
                        dropout:Boolean=true,
                        debug:Boolean=false):(Array[Array[Array[Double]]],Array[Array[Double]],Array[Array[Array[Double]]],Array[Array[Double]],Array[Array[Double]],Double)={
    /*
     * 初始化输出数据
     * */
    val layers_train_W_add_tmp:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()//每一层的参数增量w_add_tmp
    val layers_train_b_add_tmp:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层的参数增量b_add_tmp 
    val layers_train_W_hh_add_tmp:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()//每一层的参数增量W_hh_add_tmp
    val layers_train_b_hh_add_tmp:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层的参数增量h_hh_add_tmp
    val layers_train_dv:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层的局部梯度dv
    /*
     * step1 backward log_layer
     */
    var log_backward_result:(Array[Array[Double]],Array[Double],Array[Double],Double)=null
    var train_cross_entropy_result:Double=0.0
    if(is_backward_for_loglayer==true){
      if(debug) print("log layer \n") 
      log_backward_result=log_layer.backward(p_y_given_x_softmax=layer_inputs(n_layers+1), 
                                             x=layer_inputs(n_layers),
                                             y=y)
      layers_train_W_add_tmp += log_backward_result._1
      layers_train_b_add_tmp += log_backward_result._2
      train_cross_entropy_result=log_backward_result._4     
      if(debug) {print("dv=");log_backward_result._3.map(x=>print(x+"\t"));print("\n")}
    }
    /*
     * step2 backward hidden_layers
     */  
    var hidden_layers_i_train_d_v:Array[Double]=Array() 
    for(i <-(0 until n_layers).reverse){
      if(debug) print("hidden "+i+"\n")
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
                                      is_last_layer=false,                     //后面还有log_layer层结构
                                      next_layer_w=log_layer.W,                //使用log_layer的w
                                      next_layer_dv=log_backward_result._3,    //使用log_layer的d_v
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
                                      is_last_layer=true,                      //后面没有log_layer层结构了(由于i == n_layers-1所以更没有hidden了)
                                      next_layer_w=null,                       //无null
                                      next_layer_dv=null,                      //无null
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
                                    is_last_layer=false,                       //后面有hidden(由于i < n_layers-1)
                                    next_layer_w=hidden_layers(i+1).W,         //使用下一层hidden_layers(i+1)的w
                                    next_layer_dv=hidden_layers_i_train_d_v,   //使用下一层hidden_layers(i+1)的d_v
                                    dropout=dropout, 
                                    mask=if(dropout) dropout_masks(i) else Array())        
        }
        layers_train_W_add_tmp +=hidden_layers_i_train._1
        layers_train_b_add_tmp +=hidden_layers_i_train._2
        layers_train_W_hh_add_tmp +=hidden_layers_i_train._3
        layers_train_b_hh_add_tmp +=hidden_layers_i_train._4
        hidden_layers_i_train_d_v=hidden_layers_i_train._5    
        layers_train_dv+=hidden_layers_i_train_d_v
        if(debug) {print("dv=");hidden_layers_i_train_d_v.map(x=>print(x+"\t"));print("\n")}
    }
    (layers_train_W_add_tmp.reverse.toArray,layers_train_b_add_tmp.reverse.toArray,layers_train_W_hh_add_tmp.reverse.toArray,layers_train_b_hh_add_tmp.reverse.toArray,layers_train_dv.reverse.toArray,train_cross_entropy_result)
  }
  
  
  
  /**************** 
   * MLP的向前传播 
   * 一个序列的所有时间状态下,经过mlp的向前传播
   * 输入
   *   x_list 一个样本的输入x  但是此时是一个序列
   *   dropout 是否使用dropout
   *   p_dropout dropout的比例
   *     
   * 输出
   *   forward_times_result 由0->(win_times-1)时间状态下的每一个正向传播的输出
   */
  def forward(x_list:Array[Array[Double]],
              dropout:Boolean=true, 
              p_dropout:Double=0.3,debug:Boolean=false):Array[(Array[Array[Double]],Array[Array[Double]],Array[Array[Double]])]={
    /*
     * 初始化输出数据
     * */
    val forward_times_result:ArrayBuffer[(Array[Array[Double]],Array[Array[Double]],Array[Array[Double]])]=ArrayBuffer()
    /* 
     * 遍历每一个时间状态0 --> win_times-1
     */
    for(t_i <-0 until win_times){
      if(debug) print("time "+t_i+"\n")
      if(t_i==(win_times-1) && t_i==0){
        //win_size=1
        forward_times_result +=forward_one_time(x=x_list(t_i),
                                                is_first_time=true,
                                                pre_time_layers_output=null,
                                                is_forward_for_loglayer=true,
                                                dropout=dropout, 
                                                p_dropout=p_dropout,debug=debug)         
      }
      else if(t_i==0){
        //第一个时间状态t0
        forward_times_result +=forward_one_time(x=x_list(t_i),
                                                is_first_time=true,
                                                pre_time_layers_output=null,
                                                is_forward_for_loglayer=(if(RNN_structure=="full") true else false),
                                                dropout=dropout, 
                                                p_dropout=p_dropout,debug=debug)         
      }else if(t_i==(win_times-1)){
        //最后一个时间状态(win_times-1)
        forward_times_result +=forward_one_time(x=x_list(t_i),
                                                is_first_time=false,
                                                pre_time_layers_output=forward_times_result(t_i-1)._1.slice(from=1,until=(n_layers+1)),//使用上一个时间状态正向传播的结果layer_inputs的1->n_layers中hidden的输出
                                                is_forward_for_loglayer=true,//最后一个时间状态一定有log输出
                                                dropout=dropout, 
                                                p_dropout=p_dropout,debug=debug)          
      }else{
        //中间的时间状态
        forward_times_result +=forward_one_time(x=x_list(t_i),
                                                is_first_time=false,
                                                pre_time_layers_output=forward_times_result(t_i-1)._1.slice(from=1,until=(n_layers+1)),//使用上一个时间状态正向传播的结果layer_inputs的1->n_layers中hidden的输出
                                                is_forward_for_loglayer=(if(RNN_structure=="full") true else false),
                                                dropout=dropout, 
                                                p_dropout=p_dropout,debug=debug)         
      }
    }  
    forward_times_result.toArray
  }

  
  /**************** 
   * MLP的向后传播 
   * 一个序列的所有时间状态下,经过mlp的向后传播
   * 输入
   *   y_list 一个样本的输入y  但是此时是一个序列
   *                 如果RNN_structure=='full'即每次时间状态都有一个y对应        则y=多个Array[Int]的序列
   *                 如果RNN_structure=='one'即仅在最后一个时间状态才有y对应   则y=1个Array[Int]的序列
   *   dropout 是否使用dropout
   *     
   * 输出
   *   backward_times_result 由0->(win_times-1)时间状态下的每一个向后传播的输出
   */
  def backward(y_list:Array[Array[Int]],
               forward_times_result:Array[(Array[Array[Double]],Array[Array[Double]],Array[Array[Double]])],
               dropout:Boolean=true,debug:Boolean=false):Array[(Array[Array[Array[Double]]],Array[Array[Double]],Array[Array[Array[Double]]],Array[Array[Double]],Array[Array[Double]],Double)]={
    /*
     * 初始化输出数据
     * */
    val backward_times_result:ArrayBuffer[(Array[Array[Array[Double]]],Array[Array[Double]],Array[Array[Array[Double]]],Array[Array[Double]],Array[Array[Double]],Double)]=ArrayBuffer()
    /* 
     * 遍历每一个时间状态 win_times-1  --> 0
     */
    var count_i:Int=0
    for(t_i <-(0 until win_times).reverse){
      if(debug) print("time "+t_i+"\n")
      if(t_i==(win_times-1) && t_i==0){
        //win_size=1
        backward_times_result+=backward_one_time(y=y_list(t_i),
                                                 layer_inputs=forward_times_result(t_i)._1,
                                                 layer_doutputs=forward_times_result(t_i)._2,
                                                 dropout_masks=forward_times_result(t_i)._3,
                                                 is_first_time=true,
                                                 pre_time_layers_output=null,
                                                 is_last_time=true,
                                                 next_time_dv=null,
                                                 is_backward_for_loglayer=true,
                                                 dropout=dropout,debug=debug)  
      }
      else if(t_i==(win_times-1)){
        //最后一个时间状态(win_times-1)
        backward_times_result+=backward_one_time(y=(if(RNN_structure=="full") y_list(t_i) else y_list(0)),
                                                 layer_inputs=forward_times_result(t_i)._1,
                                                 layer_doutputs=forward_times_result(t_i)._2,
                                                 dropout_masks=forward_times_result(t_i)._3,
                                                 is_first_time=false,
                                                 pre_time_layers_output=forward_times_result(t_i-1)._1.slice(from=1,until=(n_layers+1)),
                                                 is_last_time=true,
                                                 next_time_dv=null,
                                                 is_backward_for_loglayer=true,
                                                 dropout=dropout,debug=debug)
      }else if(t_i==0){
        //第一个时间状态t0
        backward_times_result+=backward_one_time(y=(if(RNN_structure=="full") y_list(t_i) else null),
                                                 layer_inputs=forward_times_result(t_i)._1,
                                                 layer_doutputs=forward_times_result(t_i)._2,
                                                 dropout_masks=forward_times_result(t_i)._3,
                                                 is_first_time=true,
                                                 pre_time_layers_output=null,
                                                 is_last_time=false,
                                                 next_time_dv=backward_times_result(count_i-1)._5,
                                                 is_backward_for_loglayer=(if(RNN_structure=="full") true else false),
                                                 dropout=dropout,debug=debug)  
      }else{
        //中间的时间状态
        backward_times_result+=backward_one_time(y=(if(RNN_structure=="full") y_list(t_i) else null),
                                                 layer_inputs=forward_times_result(t_i)._1,
                                                 layer_doutputs=forward_times_result(t_i)._2,
                                                 dropout_masks=forward_times_result(t_i)._3,
                                                 is_first_time=false,
                                                 pre_time_layers_output=forward_times_result(t_i-1)._1.slice(from=1,until=(n_layers+1)),
                                                 is_last_time=false,
                                                 next_time_dv=backward_times_result(count_i-1)._5,
                                                 is_backward_for_loglayer=(if(RNN_structure=="full") true else false),
                                                 dropout=dropout,debug=debug) 
      }
      count_i+=1
    }
    backward_times_result.reverse.toArray
  }  
  
  
  /*
   * 使用Array[一串序列输入x] 来训练
   * */
  def train_batch(x_list_batch:Array[Array[Array[Double]]],
                  y_list_batch:Array[Array[Array[Int]]],
                  lr: Double,
                  batch_num_per:Double=1.0,
                  dropout:Boolean=true,
                  p_dropout:Double=0.3,
                  debug:Boolean=false){
    /*
     * 抽样数据
     * */
    //抽取样本个数
    val batch_num:Int=if(batch_num_per==1.0){
      x_list_batch.length
    }else{
      round((x_list_batch.length*batch_num_per).toFloat)//每次批量训练样本数
    }
    val rng_epooch:Random=new Random()//每次生成一个种子
    val rng_index:ArrayBuffer[Int]=ArrayBuffer();
    if(batch_num_per==1.0){
      for(i <- 0 to (batch_num-1)) rng_index += i//抽样样本的角标 
    }else{
      for(i <- 0 to (batch_num-1)) rng_index += round((rng_epooch.nextDouble()*(x_list_batch.length-1)).toFloat)//抽样样本的角标        
    }  
    /* 
     * 初始化参数增量 layers_train_W_add layers_train_b_add  layers_train_W_hh_add  layers_train_b_hh_add
     * */
    val layers_train_W_add:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()//每一层的参数增量w_add
    val layers_train_b_add:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层的参数增量b_add 
    val layers_train_W_hh_add:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()//每一层的参数增量W_hh_add
    val layers_train_b_hh_add:ArrayBuffer[Array[Double]]=ArrayBuffer()//每一层的参数增量h_hh_add
    for(i<- 0 until n_layers){
      layers_train_W_add += Array.ofDim[Double](hidden_layers(i).n_out, hidden_layers(i).n_in)
      layers_train_b_add += new Array[Double](hidden_layers(i).n_out)
      layers_train_W_hh_add += Array.ofDim[Double](hidden_layers(i).n_out, hidden_layers(i).n_out)
      layers_train_b_hh_add += new Array[Double](hidden_layers(i).n_out)      
    }
    layers_train_W_add += Array.ofDim[Double](log_layer.n_out,log_layer.n_in)
    layers_train_b_add += new Array[Double](log_layer.n_out)   
    /* 
     * 批量训练
     * */      
    var cross_entropy:Double=0.0
    for(i <- rng_index) {
      if(debug) print("样本"+i+"\n")
      //一个样本(序列)向前传播
      if(debug) print("forward:"+"\n")
      val forward_times_result=forward(x_list=x_list_batch(i),
                                       dropout=dropout, 
                                       p_dropout=p_dropout,
                                       debug=debug)
      //一个样本(序列)向后传播
      if(debug) print("backward:"+"\n")
      val backward_times_result=backward(y_list=y_list_batch(i),
                                         forward_times_result=forward_times_result,
                                         dropout=dropout,
                                         debug=debug)
      /* 
       * 更新w_add和b_add w_hh_add b_hh_add
       */
      //hidden_layers
      for(layer_i<- 0 until n_layers){
        for(t_i<-0 until win_times ){
          for(i<- 0 until hidden_layers(layer_i).n_out){
            for(j<- 0 until hidden_layers(layer_i).n_in){
              layers_train_W_add(layer_i)(i)(j) +=  backward_times_result(t_i)._1(layer_i)(i)(j)/win_times
            } 
            layers_train_b_add(layer_i)(i) +=  backward_times_result(t_i)._2(layer_i)(i)/win_times
            for(j<- 0 until hidden_layers(layer_i).n_out){
              layers_train_W_hh_add(layer_i)(i)(j) +=  backward_times_result(t_i)._3(layer_i)(i)(j)/win_times
            } 
            layers_train_b_hh_add(layer_i)(i) +=  backward_times_result(t_i)._4(layer_i)(i)/win_times
          }
        }
      }
      /*if(debug){
        print("一次样本训练后,各w的变化_add:\n")
        for(layer_i<- 0 until n_layers){
          print("hidden_"+layer_i+"的w\n")
          for(i<- 0 until hidden_layers(layer_i).n_out){
            print(hidden_layers(layer_i).W(i).mkString(sep="\t"));print("\n")
          }
          print("hidden_"+layer_i+"的w_hh\n")
          for(i<- 0 until hidden_layers(layer_i).n_out){
            print(hidden_layers(layer_i).W_hh(i).mkString(sep="\t"));print("\n")
          }          
        }
      }*/
      //log_layer
      if(RNN_structure=="full"){
        //full则每个时间都有log输出层
        for(t_i<-0 until win_times ){
          for(i<- 0 until log_layer.n_out){
            for(j<- 0 until log_layer.n_in){
              layers_train_W_add(n_layers)(i)(j) += backward_times_result(t_i)._1(n_layers)(i)(j)/win_times  
            }
            layers_train_b_add(n_layers)(i) +=  backward_times_result(t_i)._2(n_layers)(i)/win_times
          }
        }
      }else{
        //one 则仅仅在最后一层win_times-1  有 log
        for(i<- 0 until log_layer.n_out){
          for(j<- 0 until log_layer.n_in){
            layers_train_W_add(n_layers)(i)(j) += backward_times_result(win_times-1)._1(n_layers)(i)(j)/1  
          }
          layers_train_b_add(n_layers)(i) +=  backward_times_result(win_times-1)._2(n_layers)(i)/1
        }       
      }
      /*if(debug){
        print("log_layer的w\n")
        for(i<- 0 until log_layer.n_out){
          print(log_layer.W(i).mkString(sep="\t"));print("\n")
        }
      }*/       
      cross_entropy += backward_times_result(win_times-1)._6
      if(debug){
        print(backward_times_result(win_times-1)._6+"\n")
      }
    }
    /*
     * 更新w和b
     */
    for(layer_i<- 0 until n_layers){
      for(i<- 0 until hidden_layers(layer_i).n_out){
        for(j<- 0 until hidden_layers(layer_i).n_in){
          hidden_layers(layer_i).W(i)(j) +=lr*layers_train_W_add(layer_i)(i)(j)/batch_num
        }
        hidden_layers(layer_i).b(i) +=lr*layers_train_b_add(layer_i)(i)/batch_num
        for(j<- 0 until hidden_layers(layer_i).n_out){
          hidden_layers(layer_i).W_hh(i)(j) +=lr*layers_train_W_hh_add(layer_i)(i)(j)/batch_num
        }
        hidden_layers(layer_i).b_hh(i) +=lr*layers_train_b_hh_add(layer_i)(i)/batch_num          
      }
    }
    for(i<- 0 until log_layer.n_out){
      for(j<- 0 until log_layer.n_in){
        log_layer.W(i)(j) +=lr*layers_train_W_add(n_layers)(i)(j)/batch_num    
      }
      log_layer.b(i) +=lr*layers_train_b_add(n_layers)(i)/batch_num 
    }    
    print(cross_entropy/batch_num+"\n")
  }
}

object RNN {
  def test_sample_for_one(){
    /*
     * 十进制转化为0000100000类似的编码
     * trans_y_to_bit(1,8).foreach(x=>print(x+"\t"))    00000010  (8个bit输出,并且第1为=1)
     * trans_y_to_bit(0,8).foreach(x=>print(x+"\t"))    00000001  (8个bit输出,并且第0为=1)
     * */
    def trans_y_to_bit(pred_in:Int,max_size:Int):Array[Int]={
      val result:Array[Int]=new Array(max_size);
      result(pred_in)=1
      result      
    }
    /*
     * 十进制转化为二进制
     * trans_10_to_2(12,8).foreach(x=>print(x+"\t"))
     * */
    def trans_10_to_2(in_10:Int,out_sizes:Int):Array[Int]={
      val result:Array[Int]=new Array(out_sizes)
      val tmp=Integer.toBinaryString(in_10)
      //print(tmp+"\n")
      for(i<-0 until tmp.length()){
        result(out_sizes-i-1)=tmp(tmp.length()-i-1).toInt-48
      }
      result
    }    
    //Array(0.1191,0.7101,0.0012)-->Array(0,1,0)
    def trans_pred_to_bin(pred_in:Array[Double],max_size:Int):Array[Int]={
      var max_index:Int=0
      for(i <-1 until pred_in.length){
        if(pred_in(i)>pred_in(max_index)){
          max_index=i
        }
      }
      val result:Array[Int]=new Array(max_size);
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
    
    val train_X:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()
    val train_Y:ArrayBuffer[Array[Array[Int]]]=ArrayBuffer()
    val max_size=5
    val win_size =3 
    val nout=3
    train_X+=Array(8,7,1).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(1,0,0))
    train_X+=Array(8,7,2).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(1,0,0))
    train_X+=Array(8,7,3).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(1,0,0))
    train_X+=Array(8,7,4).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(1,0,0))    
    train_X+=Array(2,8,7).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(0,1,0))
    train_X+=Array(1,8,7).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(0,1,0))
    train_X+=Array(3,8,7).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(0,1,0))
    train_X+=Array(4,8,7).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(0,1,0))
    train_X+=Array(7,1,8).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(0,0,1))
    train_X+=Array(7,2,8).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(0,0,1))
    train_X+=Array(7,3,8).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(0,0,1))
    train_X+=Array(7,4,8).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); train_Y +=Array(Array(0,0,1))
    
    /*
     * trains
     * */
    val RNN_obj=new RNN(_n_in=max_size, 
          _hidden_layer_sizes=Array(50,50), 
          _n_out=nout,
          _win_times=win_size,
          _RNN_structure="one",activation="ReLU")
    var lr:Double=0.1
    for(i<-0 until 50){
      print("训练第"+i+"次:")
      RNN_obj.train_batch(x_list_batch=train_X.toArray, y_list_batch=train_Y.toArray, lr=lr,dropout=false,debug=false)  
      lr=lr*0.99
    }
    
    /*
     * test
     * */
    val test_X:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()
    test_X+=Array(8,7,5).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); //100
    test_X+=Array(5,8,7).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); //010
    test_X+=Array(7,5,8).map(x=>trans_10_to_2(x,max_size).map(x=>x.toDouble)); //001
    for(i<-0 until test_X.length){
      val forward_times_result=RNN_obj.forward(x_list=test_X(i),dropout=false)      
      val predict=forward_times_result(RNN_obj.win_times-1)._1(RNN_obj.n_layers+1)      
      print("第"+i+"个样本预测值:\n")
      print(predict.mkString(sep=","))
      print("\n") 
    }  
    /*
第0个样本预测值:
0.983921121050452,0.005235337106477452,0.010843541843070497
第1个样本预测值:
0.004881897906016043,0.8986162415031171,0.09650186059086675
第2个样本预测值:
0.010466833389995836,0.007587477800928765,0.9819456888090754
     * */
  }

  
def test_sample_for_one2(){
    /*
     * 十进制转化为0000100000类似的编码
     * trans_y_to_bit(1,8).foreach(x=>print(x+"\t"))    00000010  (8个bit输出,并且第1为=1)
     * trans_y_to_bit(0,8).foreach(x=>print(x+"\t"))    00000001  (8个bit输出,并且第0为=1)
     * */
    def trans_y_to_bit(pred_in:Int,max_size:Int):Array[Int]={
      val result:Array[Int]=new Array(max_size);
      result(pred_in)=1
      result      
    }
    /*
     * 十进制转化为二进制
     * trans_10_to_2(12,8).foreach(x=>print(x+"\t"))
     * */
    def trans_10_to_2(in_10:Int,out_sizes:Int):Array[Int]={
      val result:Array[Int]=new Array(out_sizes)
      val tmp=Integer.toBinaryString(in_10)
      //print(tmp+"\n")
      for(i<-0 until tmp.length()){
        result(out_sizes-i-1)=tmp(tmp.length()-i-1).toInt-48
      }
      result
    }    
    //Array(0.1191,0.7101,0.0012)-->Array(0,1,0)
    def trans_pred_to_bin(pred_in:Array[Double],max_size:Int):Array[Int]={
      var max_index:Int=0
      for(i <-1 until pred_in.length){
        if(pred_in(i)>pred_in(max_index)){
          max_index=i
        }
      }
      val result:Array[Int]=new Array(max_size);
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
    
    val train_X:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()
    val train_Y:ArrayBuffer[Array[Array[Int]]]=ArrayBuffer()
    val max_size=12
    val win_size =1 
    val nout=3
    train_X+=Array(Array(0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)); train_Y +=Array(Array(0,0,1))
    train_X+=Array(Array(0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0)); train_Y +=Array(Array(0,0,1))
    train_X+=Array(Array(0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0)); train_Y +=Array(Array(0,0,1))
    train_X+=Array(Array(0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0)); train_Y +=Array(Array(0,0,1))    
    train_X+=Array(Array(1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0)); train_Y +=Array(Array(1,0,0))
    train_X+=Array(Array(1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0)); train_Y +=Array(Array(1,0,0))
    train_X+=Array(Array(1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0)); train_Y +=Array(Array(1,0,0))
    train_X+=Array(Array(1.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0)); train_Y +=Array(Array(1,0,0))    
    train_X+=Array(Array(1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0)); train_Y +=Array(Array(0,1,0))
    train_X+=Array(Array(1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0)); train_Y +=Array(Array(0,1,0))
    train_X+=Array(Array(1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0)); train_Y +=Array(Array(0,1,0))
    train_X+=Array(Array(1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0)); train_Y +=Array(Array(0,1,0))
    
    /*
     * trains
     * */
    val RNN_obj=new RNN(_n_in=max_size, 
          _hidden_layer_sizes=Array(100,50), 
          _n_out=nout,
          _win_times=win_size,
          _RNN_structure="one",activation="ReLU")
    var lr:Double=0.1
    for(i<-0 until 50){
      print("训练第"+i+"次:")
      RNN_obj.train_batch(x_list_batch=train_X.toArray, y_list_batch=train_Y.toArray, lr=lr,dropout=false,debug=false)  
      //lr=lr*0.9
    }
    
    /*
     * test
     * */
    val test_X:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()
    test_X+=Array(Array(1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0)); //100
    test_X+=Array(Array(1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0)); //010
    test_X+=Array(Array(0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0)); //001
    for(i<-0 until test_X.length){
      val forward_times_result=RNN_obj.forward(x_list=test_X(i),dropout=false)      
      val predict=forward_times_result(RNN_obj.win_times-1)._1(RNN_obj.n_layers+1)      
      print("第"+i+"个样本预测值:\n")
      print(predict.mkString(sep=","))
      print("\n") 
    }  
/*
第0个样本预测值:
0.9911854283097301,0.0062315907548936384,0.002582980935376312
第1个样本预测值:
0.0023803873630450067,0.9844693012718301,0.013150311365124887
第2个样本预测值:
0.009382521034865816,0.015522780060201243,0.9750946989049329
 * */    
  }
  
  
  def test_fold_for_one(){
    //refer to http://deeplearning.net/tutorial/rnnslu.html
    
    //load fold dataset
    val (lexs,nes,labels,code2words,code2labels)=dp_utils.dataset.load_fold("D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/fold/fold_code.txt",
                        "D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/fold/fold_dict_words.txt",
                        "D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/fold/fold_dict_labels.txt")
    
    /* 
     * nv :: size of our vocabulary
     * de :: dimension of the embedding space
     * cs :: context window size
     */
    val nv=code2words.toList.length
    val de=30//或50
    val cs =5//3 or 5 or 7 
    val labels_num=code2labels.toList.length
    val nout=128//英文label最多有127个  使用00000100000形式的编码 所以才有127个 我多加1个再
    
    //转化x
    //contextwin(Array(0, 1, 2, 3, 4),3).foreach(x=>{x.foreach(x=>print(x+"\t"));print("\n")})
    //contextwin(Array(0, 1, 2, 3, 4),7).foreach(x=>{x.foreach(x=>print(x+"\t"));print("\n")})    
    def contextwin(x_list_in:Array[Int],win_in:Int):Array[Array[Int]]={
      val result:Array[Array[Int]]=Array.ofDim[Int](x_list_in.length,win_in)
      val log_x_list:Array[Int]=new Array(x_list_in.length+(win_in-1))
      for(i<-0 until log_x_list.length){
        if(i<(win_in-1)/2 || i>= (win_in-1)/2+x_list_in.length)
          log_x_list(i)= -1
        else
          log_x_list(i)=x_list_in(i- (win_in-1)/2)
      }
      //log_x_list.foreach(x=>print(x+"\t"));print("\n")
      for(i<-0 until x_list_in.length){
        result(i)=log_x_list.slice(from=i, until=i+win_in)
      }
      result
    }
    
    /*定义uniform均值分布的随机数产生函数
   * max 均匀分布的max
   * min 均匀分布的min
   * gen_nums  随机数生成个数
   * rng 随机种子
   * */
    def random_uniform(min: Double, max: Double,gen_nums:Int,rng:Random=new Random(1234)): Array[Double] ={
      val result:Array[Double]=new Array(gen_nums);
      for(i <- 0 until gen_nums){
        result(i)=rng.nextDouble()*(max - min) + min
      }
      result
    }
    
    /*
     * Word embeddings
     * */
    val embeddings_0 = random_uniform(-1,1,(nv+1)*de)
    val embeddings:Array[Array[Double]]=Array.ofDim[Double](nv+1,de)
    for(i <-0 until nv+1){
      for(j <-0 until de){  
        embeddings(i)(j)=embeddings_0(i*de+j)
      }
    }
    def trans_word_to_embeddings(in_word_idx:Int):Array[Double]={
      if(in_word_idx== -1){
        embeddings(0)
      }else{
        embeddings(in_word_idx+1)  
      }
    }
    //print(embeddings.foreach(x=>{x.foreach(y=>print(y+"\t"));print("\n")}))
    
    /*
     * 十进制转化为0000100000类似的编码
     * trans_y_to_bit(1,8).foreach(x=>print(x+"\t"))    00000010  (8个bit输出,并且第1为=1)
     * trans_y_to_bit(0,8).foreach(x=>print(x+"\t"))    00000001  (8个bit输出,并且第0为=1)
     * */
    def trans_y_to_bit(pred_in:Int,max_size:Int):Array[Int]={
      val result:Array[Int]=new Array(max_size);
      result(pred_in)=1
      result      
    }
    
    /*
     * 转化数据集
     * */
    val dataset_X:ArrayBuffer[Array[Array[Double]]]=ArrayBuffer()
    val dataset_Y:ArrayBuffer[Array[Array[Int]]]=ArrayBuffer()
    for(i<-0 until lexs.length){
      val tmp=contextwin(lexs(i),cs)
      for(j<-0 until tmp.length){
        if(labels(i)(j)==0){
        }else{
          var tmp_X:ArrayBuffer[Array[Double]]=ArrayBuffer()
          for(k<-0 until tmp(j).length){
            tmp_X+=trans_word_to_embeddings(tmp(j)(k))  
          } 
          dataset_X+=tmp_X.toArray
          dataset_Y+=Array(trans_y_to_bit(pred_in=labels(i)(j),max_size=nout))//max is 127
        }
      }
    }
    /*dataset_X(10).foreach(x=>print(x+"\t"));print("\n")
    print(dataset_X(10).length+"\n")
    dataset_Y(10).foreach(x=>print(x+"\t"));print("\n")
    print(dataset_Y(10).length+"\n")
    print(dataset_X.length)*/
    
    
    /*
     * 测试集 训练集的分割
     */
    val dataset_nums:Int=dataset_X.length
    val dataset_train_nums:Int=round(dataset_nums*0.65).toInt
    val dataset_test_nums:Int=round(dataset_nums*0.3).toInt
    //print(dataset_nums+"\t"+dataset_train_nums+"\t"+dataset_test_nums)  
    val train_X=dataset_X.slice(from=0, until=dataset_train_nums).toArray
    val train_Y=dataset_Y.slice(from=0, until=dataset_train_nums).toArray
    val test_X=dataset_X.slice(from=dataset_train_nums, until=dataset_train_nums+dataset_test_nums).toArray
    val test_Y=dataset_Y.slice(from=dataset_train_nums, until=dataset_train_nums+dataset_test_nums).toArray
    
    
    /*
     * trains
     * */
    val RNN_obj=new RNN(_n_in=de, 
          _hidden_layer_sizes=Array(100,200,100), 
          _n_out=nout,
          _win_times=cs,
          _RNN_structure="one",activation="ReLU")
    var lr:Double=0.1
    for(i<-0 until 300){
      print("训练第"+i+"次:")
      RNN_obj.train_batch(x_list_batch=train_X, y_list_batch=train_Y, lr=lr, batch_num_per=0.1, dropout=false,debug=false)  
      lr=lr*0.99
    }
    
    /*
     * test
     * */
    //Array(0.1191,0.7101,0.0012)-->Array(0,1,0)
    def trans_pred_to_bin(pred_in:Array[Double],max_size:Int):Array[Int]={
      var max_index:Int=0
      for(i <-1 until pred_in.length){
        if(pred_in(i)>pred_in(max_index)){
          max_index=i
        }
      }
      val result:Array[Int]=new Array(max_size);
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
    var pred_right_nums:Int=0
    var pred_result:scala.collection.mutable.Map[Int,(Int,Int,Double)]=scala.collection.mutable.Map()//可变映射
    //val test_N=test_X.length
    val test_N=10000
    for(i<-0 until test_N){
      val forward_times_result=RNN_obj.forward(x_list=test_X(i),dropout=false)      
      val predict=forward_times_result(RNN_obj.win_times-1)._1(RNN_obj.n_layers+1)
      print("第"+i+"个样本实际值:\n")
      print(test_Y(i)(0).mkString(sep=","))
      print("\n")        
      print("第"+i+"个样本预测值:\n")
      print(predict.mkString(sep=","))
      print("\n") 
      if(!pred_result.contains( key=trans_bin_to_int(test_Y(i)(0)))){
        pred_result += (trans_bin_to_int(test_Y(i)(0))->(0,0,0.0))
      }      
      if(trans_bin_to_int(trans_pred_to_bin(predict,nout))==trans_bin_to_int(test_Y(i)(0))){
        pred_right_nums +=1
        pred_result(trans_bin_to_int(test_Y(i)(0))) = (pred_result(trans_bin_to_int(test_Y(i)(0)))._1+1,pred_result(trans_bin_to_int(test_Y(i)(0)))._2+1,(pred_result(trans_bin_to_int(test_Y(i)(0)))._1+1).toDouble/(pred_result(trans_bin_to_int(test_Y(i)(0)))._2+1).toDouble)
      }
      pred_result(trans_bin_to_int(test_Y(i)(0))) = (pred_result(trans_bin_to_int(test_Y(i)(0)))._1,pred_result(trans_bin_to_int(test_Y(i)(0)))._2+1,(pred_result(trans_bin_to_int(test_Y(i)(0)))._1).toDouble/(pred_result(trans_bin_to_int(test_Y(i)(0)))._2+1).toDouble)
    }
    println(pred_right_nums.toDouble/(test_N.toDouble))
    for(i<-pred_result.keys){
      if(pred_result(i)._2>=100){
        print(i+":\n"+pred_result(i)._1+"\t"+pred_result(i)._2+"\t"+pred_result(i)._3+"\n")  
      }
    }
    
  }
  def main(args: Array[String]) {
    test_fold_for_one()//由于label=0的情况太多,故删除了---59%--不好
    //test_sample_for_one()//ok
    //test_sample_for_one2()//ok
  }  
}