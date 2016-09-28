package dp_process


  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

/*
 * 仅仅包括1层hidden的RNN
 * |<--------------------------------------win--------------------------------------------------------------->| 
 * |                                                                                                          | 
 * | t=0                                 |          t=1                               |   t=2....t=(win-1)    |
 * |                                     |                                            |                       |
 * | output_layer(nout个神经元)             |          output_layer(nout个神经元)           |                       |                 
 * |       /\                            |              /\                            |                       |
 * |       ||w_hidden2out                |              ||w_hidden2out                |                       |
 * |       ||b_hidden2out         w_hidden2hidden       ||b_hidden2out          w_hidden2hidden               |
 * | hidden_layer(nhidden个神经元) ==================> hidden_layer(nhidden个神经元)==================>    MLP      |
 * |       /\                     b_hidden2hidden       /\                      b_hidden2hidden               |
 * |       ||                            |              ||                            |                       |
 * |       ||w_in2hidden                 |              ||w_in2hidden                 |                       |
 * |       ||b_in2hidden                 |              ||b_in2hidden                 |                       |
 * | input(nin个神经元)                     |          input(nin个神经元)                   |                       |
 * */
class RNN_one_hiddenlayer(_win:Int,
          _nin:Int,
          _nhidden:Int,
          _nout:Int){
  
  val win:Int=_win         //RNN时长=MLP的个数
  val nin:Int=_nin         //MLP的输入个数
  val nhidden:Int=_nhidden //MLP的隐藏层个数
  val nout:Int=_nout       //MLP的输出个数
  val w_in2hidden:Array[Array[Double]]=Array.ofDim[Double](nhidden,nin)         //MLP内输入层到隐藏层的w系数
  val b_in2hidden:Array[Double]=new Array(nhidden)                              //MLP内输入层到隐藏层的b系数
  val w_hidden2hidden:Array[Array[Double]]=Array.ofDim[Double](nhidden,nhidden) //MLP内隐藏层到隐藏层的w系数
  val b_hidden2hidden:Array[Double]=new Array(nhidden)                          //MLP内隐藏层到隐藏层的w系数
  val w_hidden2out:Array[Array[Double]]=Array.ofDim[Double](nout,nhidden)       //MLP内隐藏层到输出层的w系数
  val b_hidden2out:Array[Double]=new Array(nout)                                //MLP内隐藏层到输出层的w系数
  
  /*
   * 向前擦传播
   * */
  def forward(x:Array[Double]):Array[Double]={
    Array()
  }
}


object RNN_one_hiddenlayer {
  
}