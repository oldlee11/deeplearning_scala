package dp_process_parallel

import scala.util.Random
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array
import scala.math

import java.io.File; 
import java.io._
import scala.io.Source


/*
 * ConvPoolLayer=ConvLayer=>Max_PoolLayer
 * input_size_in:conv卷积层输入
 * n_kernel_in:conv卷积层的卷积核个数
 * kernel_size_in:conv卷积层的卷积核长和高
 * pool_size_in:maxpool池化层的核长和高
 * init_a_in 用于conv卷积层的初始化
 * _W 初始化conv卷积层的w系数
 * _b 初始化conv卷积层的b系数
 * activation  conv卷积层的非线性处理函数
 * */
class ConvPoolLayer_parallel(input_size_in:(Int,Int),
          n_kernel_in:Int,
          kernel_size_in:(Int,Int),
          pool_size_in:(Int,Int),
          _W:Array[Array[Array[Array[Double]]]]=null,
          _b:Array[Double]=null,
          n_channel_in:Int=3,
          _rng: Random=null,
          activation:String="ReLU") {
  
  var rng:Random=if(_rng == null) new Random(1234) else _rng
  
  //用于初始化conv成参数
  //参考lisa lab和yusugomori
  val f_in_tmp:Int  = n_channel_in * kernel_size_in._1 * kernel_size_in._2
  val f_out_tmp:Int = (n_kernel_in * kernel_size_in._1 * kernel_size_in._2)/(pool_size_in._1*pool_size_in._2)
  val init_a:Double=math.sqrt(6.0/(f_in_tmp + f_out_tmp)) 
  
  //其它全局变量
  val input_size:(Int,Int)=input_size_in
  val n_kernel:Int=n_kernel_in
  val kernel_size:(Int,Int)=kernel_size_in
  val n_channel:Int=n_channel_in  
  val conv_s0:Int = input_size._1 - kernel_size._1 + 1//卷积层输出的size
  val conv_s1:Int = input_size._2 - kernel_size._2 + 1//卷积层输出的size
  //(conv_s0,conv_s1)作为maxpool的input_size
  val pool_size:(Int,Int)=pool_size_in
  val pool_s0:Int = math.floor((conv_s0.toDouble)/(pool_size._1.toDouble)).toInt//向下取整 //池化层输出的size
  val pool_s1:Int = math.floor((conv_s1.toDouble)/(pool_size._2.toDouble)).toInt//向下取整 //池化层输出的size

  /*
   * 建立参数W和b并初始化
   */
  var W:Array[Array[Array[Array[Double]]]]=Array.ofDim[Double](n_kernel,n_channel,kernel_size._1, kernel_size._2)
  var W_add:Array[Array[Array[Array[Double]]]]=Array.ofDim[Double](n_kernel,n_channel,kernel_size._1, kernel_size._2)
  var b:Array[Double]=new Array[Double](n_kernel)  
  var b_add:Array[Double]=new Array[Double](n_kernel)    
  def init_w_module(init_a:Double){
    print(init_a+"\n")//debug
    for(i <- 0 until n_kernel){
      for(j <- 0 until n_channel){
        for(m <- 0 until kernel_size._1){
          for(n <- 0 until kernel_size._2){
            W(i)(j)(m)(n) = uniform(-init_a, init_a)  
          }
        }
      }
      b(i)=0.0 
    }
  }
  init_w_module(init_a);
  if(_W == null) {
  } else {
    W = _W
  }
  if(_b == null) {
  } else {
    b = _b
  }  
  
  
  /*
   * conv->maxpool向前算法
   * 输入一个样本x
   * 最后输出(convolved_input,pooled_input,max_index_x) 用于向后传播算法
   * */
  def cnn_forward(x:Array[Array[Array[Double]]],
                            debug_time:Boolean=false):(Array[Array[Array[Double]]],Array[Array[Array[Double]]],Array[Array[Array[(Int,Int)]]])={
    /*
     * Conv卷积层的向前算法
     * */
    if(debug_time) deal_begin_time()
    val convolved_input:Array[Array[Array[Double]]]=Array.ofDim[Double](n_kernel,conv_s0,conv_s1)//convolved_input=一个channel数据,在经过卷积网络处理后(n_kernel个核) 输出数据的size=n_kernel个 s0*s1
    val activated_input:Array[Array[Array[Double]]]=Array.ofDim[Double](n_kernel,conv_s0,conv_s1)//activated_input=convolved_input经过activation_fun函数处理的输出  
    for(k <- 0 until n_kernel){
      for(i <- 0 until conv_s0){ 
        for(j <- 0 until conv_s1){
          convolved_input(k)(i)(j)=0.0//情空
          for(c <- 0 until n_channel){
            for(s <- 0 until kernel_size._1){
              for(t <- 0 until kernel_size._2){
                //做相关运算  核内的元素和输入样本x的元素对应相乘后计算求和
                //实现了matlab的 convolved_input=convn(x,rot180(W),'vaild')
                convolved_input(k)(i)(j) += W(k)(c)(s)(t) * x(c)(i+s)(j+t)
              }
            }
          }
          convolved_input(k)(i)(j)=convolved_input(k)(i)(j) + b(k)
          //对每个输出经过activation_fun处理
          activated_input(k)(i)(j) = activation_fun(convolved_input(k)(i)(j))
        }
      }
    }
    if(debug_time) deal_end_time("ConvLayer_forward")
    /*
     * max池化层的向前算法
     * */
    if(debug_time) deal_begin_time()
    val max_index_x:Array[Array[Array[(Int,Int)]]]=Array.ofDim[(Int,Int)](n_kernel,pool_s0,pool_s1)//用于记录输出的max 对应输入的坐标
    val pooled_input:Array[Array[Array[Double]]]=Array.ofDim[Double](n_kernel,pool_s0,pool_s1)//把输入经过max池化处理的数据输出
    var max_index_1:Int=0;
    var max_index_2:Int=0;
    var max_tmp:Double=0.0;
    var x_index_1:Int=0;
    var x_index_2:Int=0;
    for(k <- 0 until n_kernel){      
      for(i <- 0 until pool_s0){ 
        for(j <- 0 until pool_s1){           
          for(s <- 0 until pool_size._1){
            for(t <- 0 until pool_size._2){    
              x_index_1=pool_size._1*i+s
              x_index_2=pool_size._2*j+t
              if(s==0 && t==0){
                max_tmp=activated_input(k)(x_index_1)(x_index_2)
                max_index_1=x_index_1
                max_index_2=x_index_2
              }
              if(max_tmp<activated_input(k)(x_index_1)(x_index_2)){
                max_tmp=activated_input(k)(x_index_1)(x_index_2)//取pool_size._1 * pool_size._2内的最大值
                max_index_1=x_index_1
                max_index_2=x_index_2               
              }
            }      
          }
          pooled_input(k)(i)(j)=max_tmp// 等价于 pooled_input(k)(i)(j)=x(k)(max_index_1)(max_index_2)
          max_index_x(k)(i)(j)=(max_index_1,max_index_2) //标识出输入中那个是max
        }
      }
    }  
    if(debug_time) deal_end_time("Max_PoolLayer_forward")
    (convolved_input,pooled_input,max_index_x)
  }
  
  /*
   * 下一层的参数
   * next_layer_type 下一层的类型=hidden和ConvLayer
   * 如果next_layer_type=="ConvLayer",则:
   *   next_layer_convlayer_n_channel 下一层的n_channel
   *   next_layer_convlayer_n_kernel  下一层的n_kernel
   *   next_layer_convlayer_kernel_size 下一层的kernel_size
   *   next_layer_convlayer_w         下一层的W
   *   next_layer_convlayer_dv        下一层的d_v
   * 如果next_layer_type=="hidden",则:
   *   next_layer_hidden_n_in         下一层的n_in
   *   next_layer_hidden_n_out        下一层的n_out
   *   next_layer_hidden_w            下一层的W
   *   next_layer_hidden_dv           下一层的d_v
   *     
   * 本层的参数
   * poollayer_max_index_x            本层的pool赤化层的max_index_x
   * convlayer_convolved_input        本层的卷积层的convolved_input
   * 
   * 最后输出 conv卷积层参数的一次样本训练后的变化 (W_add_tmp,b_add_tmp,conv_dv)
   * 
   * */
  def cnn_backward(x:Array[Array[Array[Double]]],
                   next_layer_type:String,
                   next_layer_convlayer_n_channel:Int=0,
                   next_layer_convlayer_n_kernel:Int=0,
                   next_layer_convlayer_kernel_size:(Int,Int)=null,
                   next_layer_convlayer_w:Array[Array[Array[Array[Double]]]]=null,
                   next_layer_convlayer_dv:Array[Array[Array[Double]]]=null,
                   next_layer_hidden_n_in:Int=0,
                   next_layer_hidden_n_out:Int=0,
                   next_layer_hidden_w:Array[Array[Double]]=null,                   
                   next_layer_hidden_dv:Array[Double]=null,
                   poollayer_max_index_x:Array[Array[Array[(Int,Int)]]]=null,
                   convlayer_convolved_input:Array[Array[Array[Double]]]=null,
                   debug_time:Boolean=false):(Array[Array[Array[Array[Double]]]],Array[Double],Array[Array[Array[Double]]])={
    /*
     * 池化层的向后算法
     * */
    val pool_dv:Array[Array[Array[Double]]]=Array.ofDim[Double](n_kernel,pool_s0,pool_s1)//maxpool层的dv 清空为0.0
    if(next_layer_type=="ConvLayer"){
      if(debug_time) deal_begin_time()
      var tmp1:Int=0
      var tmp2:Int=0    
      for (c<- 0 until next_layer_convlayer_n_channel){
        for(tmp1<- 0 until next_layer_convlayer_dv(0).length){
          for(tmp2<- 0 until next_layer_convlayer_dv(0)(0).length){
            for(k <- 0 until next_layer_convlayer_n_kernel){
              for(s <- 0 until next_layer_convlayer_kernel_size._1){
                for(t <- 0 until next_layer_convlayer_kernel_size._2){
                  //相当于MATLAB中的convn(next_layer.d_v,next_layer.W,'full'),即完成了2维的卷积运算
                  pool_dv(c)(tmp1+s)(tmp2+t) += next_layer_convlayer_dv(k)(tmp1)(tmp2) * next_layer_convlayer_w(k)(c)(s)(t)
                  //由于是max 所以没有 *dactivation_fun(input) 
                }  
              }
            }
          }  
        }
      } 
      if(debug_time) deal_end_time("Max_PoolLayer_backward1")
    }else if(next_layer_type=="hidden"){
      if(debug_time) deal_begin_time()
      //使用下一层的d_v累加计算出本层的d_v,但是是打平后的
      val flatten_size:Int=next_layer_hidden_n_in
      val d_v_flatten:Array[Double]=new Array(flatten_size)//内部为0.0 每次清零
      for(j <-0 until flatten_size){
        for(i <-0 until next_layer_hidden_n_out){
          d_v_flatten(j) = d_v_flatten(j) + next_layer_hidden_w(i)(j) * next_layer_hidden_dv(i)
        }
      } 
      //把打平后的d_v变为k,i,j
      var index:Int=0
      for(k <- 0 until n_kernel){
        for(i <- 0 until pool_s0){ 
          for(j <- 0 until pool_s1){ 
            pool_dv(k)(i)(j)=d_v_flatten(index)//由于是max 所以没有 *dactivation_fun(input)
            index +=1
          }
        }
      }   
      if(debug_time) deal_end_time("Max_PoolLayer_backward2")
    }else{
      return (Array(),Array(),Array())
    }
    /*
     * 卷积层的向后算法
     * */ 
    if(debug_time) deal_begin_time()
    //计算局部梯度
    val conv_dv:Array[Array[Array[Double]]]=Array.ofDim[Double](n_kernel,conv_s0,conv_s1)//卷积层的dv 清空为0.0
    for(k <- 0 until n_kernel){
      //遍历经过某一个池化层后输出样本矩阵的长度
      for(i <- 0 until pool_s0){ 
        //遍历经过某一个池化层后输出样本矩阵的高度
        for(j <- 0 until pool_s1){
          //参考 《CNN的反向求导及联系.pdf》中的问题2
          conv_dv(k)(poollayer_max_index_x(k)(i)(j)._1)(poollayer_max_index_x(k)(i)(j)._2)=pool_dv(k)(i)(j)*dactivation_fun(convlayer_convolved_input(k)(poollayer_max_index_x(k)(i)(j)._1)(poollayer_max_index_x(k)(i)(j)._2))          
        }
      }
    }
    //计算参数的更新大小    
    var W_add_tmp:Array[Array[Array[Array[Double]]]]=Array.ofDim[Double](n_kernel,n_channel,kernel_size._1, kernel_size._2)//W的更新大小
    var b_add_tmp:Array[Double]=new Array[Double](n_kernel)//b的更新大小    
    for(k <- 0 until n_kernel){
      for(i <- 0 until conv_s0){ 
        for(j <- 0 until conv_s1){
          b_add_tmp(k) +=conv_dv(k)(i)(j)*1.0
          for(c <- 0 until n_channel){
            for(s <- 0 until kernel_size._1){
              for(t <- 0 until kernel_size._2){
                //应该实现matlab的 W_add_tmp=convn(x,rot180(d_v),'valid'),//实际就是卷积层的正向传播,参考convolve_forward的实现
                W_add_tmp(k)(c)(s)(t) += x(c)(i+s)(j+t)*conv_dv(k)(i)(j) 
              }
            }
          }          
        }
      }
    }  
    /* debug
    x.foreach { x => print("x:\n");x.foreach { x => x.foreach { x => print(x+"\t") };print("\n") } }//debug
    conv_dv.foreach { x => print("d_v:\n");x.foreach { x => x.foreach { x => print(x+"\t") };print("\n") } }//debug
    print("W_add_0_0:\n")//debug
    W_add_tmp(0)(0).foreach { x => x.foreach { x => print(x+"\t") };print("\n") }//debug
    */
    /*
     * 更新参数
     * 在后续过程做
     * */
    if(debug_time) deal_end_time("ConvLayer_backward")
    (W_add_tmp,b_add_tmp,conv_dv)  
  }
  
  
  /*
   * 函数
   */  
    def uniform(min: Double, max: Double): Double = rng.nextDouble() * (max - min) + min
    def activation_fun(linear_output:Double):Double={
      if(activation=="sigmoid"){
        dp_process.utils.sigmoid(linear_output)  
      }else if(activation=="tanh"){
        dp_process.utils.tanh(linear_output)
      }else if(activation=="ReLU") {
        dp_process.utils.ReLU(linear_output)
      }else if(activation=="linear"){
        linear_output
      }else{
        0.0
      }
    }
    def dactivation_fun(linear_output:Double):Double={
      if(activation=="sigmoid"){
        dp_process.utils.dsigmoid(linear_output)  
      }else if(activation=="tanh"){
        dp_process.utils.dtanh(linear_output)
      }else if(activation=="ReLU") {
        dp_process.utils.dReLU(linear_output)
      }else if(activation=="linear"){
        1.0
      }else{
        0.0
      }
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


/*test*/
object ConvPoolLayer_parallel {
  def main(args: Array[String]) {

    //ok
    //数据案例使用 《CNN_forward.jpg》
    print("test for ConvLayer forward & Max_PoolLayer forward:\n")
    print("step1: test for ConvLayer forward:\n")
    var init_w:Array[Array[Array[Array[Double]]]]=Array(
        Array(
            Array(
               Array(-1, 0,-1),   
               Array( 1,-1,-1),
               Array(-1, 0,-1)
            ),  
            Array(
               Array( 1, 0, 1),   
               Array(-1, 1, 0),
               Array( 0,-1,-1)                
            ), 
            Array(
               Array(-1,-1,-1),   
               Array(-1, 0, 0),
               Array(-1,-1, 1)                  
            ) 
        ),
        Array(
            Array(
               Array( 0, 0, 0),   
               Array(-1,-1, 1),
               Array( 0,-1,-1)                
            ),  
            Array(
               Array( 1, 1, 1),   
               Array( 0,-1, 1),
               Array( 0, 0,-1)                   
            ), 
            Array(
               Array( 0, 0,-1),   
               Array( 0, 0, 1),
               Array( 0,-1, 1)                   
            )             
        )
    );
    var init_b:Array[Double]=Array(1,0);
    var ConvPoolLayer_obj_test:ConvPoolLayer_parallel =new ConvPoolLayer_parallel(input_size_in=(7,7),
                                                                n_kernel_in=2,
                                                                kernel_size_in=(3,3),
                                                                pool_size_in=(2,2),
                                                                _W=init_w,
                                                                _b=init_b,
                                                                n_channel_in=3,
                                                                activation="sigmoid")
    var x_test:Array[Array[Array[Double]]]=Array(
      Array(
            Array(0,0,0,0,0,0,0),
            Array(0,2,2,2,2,2,0),
            Array(0,2,1,2,2,2,0),
            Array(0,1,0,0,0,1,0),
            Array(0,0,1,1,2,2,0),
            Array(0,1,1,2,1,0,0),
            Array(0,0,0,0,0,0,0)
      ),   
      Array(
            Array(0,0,0,0,0,0,0),
            Array(0,2,2,1,1,0,0),
            Array(0,0,0,0,0,2,0),
            Array(0,1,2,2,2,2,0),
            Array(0,2,1,1,1,1,0),
            Array(0,2,1,2,0,2,0),
            Array(0,0,0,0,0,0,0)          
      ), 
      Array(
            Array(0,0,0,0,0,0,0),
            Array(0,2,2,1,1,0,0),
            Array(0,0,1,0,1,2,0),
            Array(0,2,1,0,1,0,0),
            Array(0,1,1,0,1,0,0),
            Array(0,0,2,1,2,2,0),
            Array(0,0,0,0,0,0,0)           
      )
    )
    val test_result1=ConvPoolLayer_obj_test.cnn_forward(x=x_test)
    test_result1._1.foreach { x => print("convolved_input_i:\n");x.foreach { x => x.foreach { x => print(x+"\t") } ;print("\n")}}    
    /*
convolved_input_i:
-1.0	-8.0	-7.0	-7.0	-8.0	
-10.0	-14.0	-12.0	-12.0	-3.0	
-5.0	-9.0	-10.0	-11.0	-10.0	
-1.0	-11.0	-5.0	-5.0	-6.0	
-1.0	-2.0	-5.0	-3.0	0.0	
convolved_input_i:
0.0	-6.0	-4.0	-8.0	-8.0	
-2.0	0.0	2.0	0.0	-6.0	
-2.0	-5.0	-3.0	-5.0	-3.0	
2.0	-1.0	4.0	2.0	-3.0	
3.0	6.0	0.0	4.0	-1.0	
     * */
    print("tesp2:test for Max_PoolLayer forward:\n")
    test_result1._2.foreach { x => print("pooled_input_i:\n");x.foreach { x => x.foreach { x => print(x+"\t") } ;print("\n")}}
/*
pooled_input_i:
0.2689414213699951	9.110511944006456E-4	
0.2689414213699951	0.006692850924284857	
pooled_input_i:
0.5	                0.8807970779778823	
0.8807970779778823	0.9820137900379085
 * */    
    test_result1._3.foreach { x => print("max_index_x_i:\n");x.foreach { x => x.foreach { x => print("("+x._1+","+x._2+")\t") } ;print("\n")}}
/*
max_index_x_i:
(0,0)	(0,2)	
(3,0)	(3,2)	
max_index_x_i:
(0,0)	(1,2)	
(3,0)	(3,2)	
 * */
    
    
    
    print("step3: test for Max_PoolLayer backward_1(maxpool的下一层是卷积层conv) :\n")
    
    
    
  }  
}