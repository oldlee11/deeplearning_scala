package dp_process_breeze


import scala.util.Random
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

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
class ConvPoolLayer_breeze(input_size_in:(Int,Int),
                           n_kernel_in:Int,
                           kernel_size_in:(Int,Int),
                           pool_size_in:(Int,Int),
                           _W:DenseMatrix[DenseMatrix[Double]]=null,
                           _b:DenseVector[Double]=null,
                           n_channel_in:Int=3,
                           _rng: Random=null,
                           activation:String="ReLU") {
  
  /**********************
   ****   init       ****
   **********************/
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
  //建立参数W和b并初始化
  var W:DenseMatrix[DenseMatrix[Double]]=matrix_utils.build_4d_matrix(size_in=(n_kernel,n_channel,kernel_size._1, kernel_size._2))
  var W_add:DenseMatrix[DenseMatrix[Double]]=matrix_utils.build_4d_matrix(size_in=(n_kernel,n_channel,kernel_size._1, kernel_size._2))
  var b:DenseVector[Double]=matrix_utils.build_1d_matrix(n_kernel)  
  var b_add:DenseVector[Double]=matrix_utils.build_1d_matrix(n_kernel)     
  def init_w_module(init_a:Double){
    print(init_a+"\n")//debug
    for(i <- 0 until n_kernel){
      for(j <- 0 until n_channel){
        for(m <- 0 until kernel_size._1){
          for(n <- 0 until kernel_size._2){
            W(i,j)(m,n) = uniform(-init_a, init_a)  
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
  

  /************************
   * 向前算法
   * conv->maxpool向前算法
   * 输入一个样本x
   * 最后输出(convolved_input,pooled_input,max_index_x) 用于向后传播算法
   ************************/
  def cnn_forward(x:DenseVector[DenseMatrix[Double]],
                  debug_time:Boolean=false):(DenseVector[DenseMatrix[Double]],DenseVector[DenseMatrix[Double]],DenseVector[DenseMatrix[Double]])={
     /*
     * Conv卷积层的向前算法
     * */
    if(debug_time) deal_begin_time()
    val convolved_input:DenseVector[DenseMatrix[Double]]=matrix_utils.build_3d_matrix((n_kernel,conv_s0,conv_s1))//convolved_input=一个channel数据,在经过卷积网络处理后(n_kernel个核) 输出数据的size=n_kernel个 s0*s1
    val activated_input:DenseVector[DenseMatrix[Double]]=matrix_utils.build_3d_matrix((n_kernel,conv_s0,conv_s1))//activated_input=convolved_input经过activation_fun函数处理的输出  
    for(k <- 0 until n_kernel){
      for(c <- 0 until n_channel){
        convolved_input(k)=convolved_input(k)+convn_breeze.cor2d(x(c),W(k,c),"valid")
      }
      convolved_input(k)=convolved_input(k)+b(k)//convolved_input(k)矩阵内的每一个元素都+b(k)
      //对每个输出经过activation_fun处理
      activated_input(k)= convolved_input(k).map(x=>activation_fun(x))//convolved_input(k)矩阵内的每个元素都activation_fun处理
    }
    if(debug_time) deal_end_time("ConvLayer_forward")
    
    /*
     * max池化层的向前算法------------没有优化？？？？？？？？？？？
     * */
    if(debug_time) deal_begin_time()
    val pooled_input_maxvalue_zero:DenseVector[DenseMatrix[Double]]=matrix_utils.build_3d_matrix((n_kernel,conv_s0,conv_s1))//把activated_input中每个poolsize内的max值保存,非max值=0.0（不是记录对应输入的坐标）
    val pooled_input:DenseVector[DenseMatrix[Double]]=matrix_utils.build_3d_matrix((n_kernel,pool_s0,pool_s1))//把输入经过max池化处理的数据输出
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
                max_tmp=activated_input(k)(x_index_1,x_index_2)
                max_index_1=x_index_1
                max_index_2=x_index_2
              }
              if(max_tmp<activated_input(k)(x_index_1,x_index_2)){
                max_tmp=activated_input(k)(x_index_1,x_index_2)//取pool_size._1 * pool_size._2内的最大值
                max_index_1=x_index_1
                max_index_2=x_index_2               
              }
            }      
          }
          pooled_input(k)(i,j)=max_tmp// 等价于 pooled_input(k)(i)(j)=x(k)(max_index_1)(max_index_2)
          pooled_input_maxvalue_zero(k)(max_index_1,max_index_2)=max_tmp //标识出输入中那个是max
        }
      }
    }  
    if(debug_time) deal_end_time("Max_PoolLayer_forward")
    (convolved_input,pooled_input,pooled_input_maxvalue_zero)
  }  

  
  
  /***************************************************
   * 向后算法
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
   * poollayer_maxvalue_zero          本层的pool赤化层的pooled_input_maxvalue_zero
   * convlayer_convolved_input        本层的卷积层的convolved_input
   * 
   * 最后输出 conv卷积层参数的一次样本训练后的变化 (W_add_tmp,b_add_tmp,conv_dv)
   * 
   *******************************************************/
  /*def cnn_backward(x:DenseVector[DenseMatrix[Double]],
                   next_layer_type:String,
                   next_layer_convlayer_n_channel:Int=0,
                   next_layer_convlayer_n_kernel:Int=0,
                   next_layer_convlayer_kernel_size:(Int,Int)=null,
                   next_layer_convlayer_w:DenseMatrix[DenseMatrix[Double]]=null,
                   next_layer_convlayer_dv:DenseVector[DenseMatrix[Double]]=null,
                   next_layer_hidden_n_in:Int=0,
                   next_layer_hidden_n_out:Int=0,
                   next_layer_hidden_w:DenseMatrix[Double]=null,                   
                   next_layer_hidden_dv:DenseVector[Double]=null,
                   poollayer_maxvalue_zero:DenseVector[DenseMatrix[Double]]=null,
                   convlayer_convolved_input:DenseVector[DenseMatrix[Double]]=null,
                   debug_time:Boolean=false):(DenseMatrix[DenseMatrix[Double]],DenseVector[Double],DenseVector[DenseMatrix[Double]])={
    
  }*/
  
  
  /**********************
   ****   函数                             ****
   **********************/  
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
    
    
  /**********************
   *  for time debug
   ***********************/
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
object ConvPoolLayer_breeze {
  def main(args: Array[String]) {
   //ok
    //数据案例使用 《CNN_forward.jpg》
    print("test for ConvLayer forward & Max_PoolLayer forward:\n")
    print("step1: test for ConvLayer forward:\n")
    val init_w_0:Array[Array[Array[Array[Double]]]]=Array(
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
    val init_w=matrix_utils.trans_array2breeze_4d(init_w_0)
    var init_b:DenseVector[Double]=new DenseVector(Array(1.0,0.0));
    var ConvPoolLayer_obj_test:ConvPoolLayer_breeze =new ConvPoolLayer_breeze(input_size_in=(7,7),
                                                                n_kernel_in=2,
                                                                kernel_size_in=(3,3),
                                                                pool_size_in=(2,2),
                                                                _W=init_w,
                                                                _b=init_b,
                                                                n_channel_in=3,
                                                                activation="sigmoid")
    val x_test_0:Array[Array[Array[Double]]]=Array(
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
    val x_test=matrix_utils.trans_array2breeze_3d(x_test_0)
    val test_result1=ConvPoolLayer_obj_test.cnn_forward(x=x_test)
    for(i<-0 until test_result1._1.length){
      print("convolved_input_"+i+":\n")
      for(j<-0 until test_result1._1(i).rows){
        for(k<-0 until test_result1._1(i).cols){
          print(test_result1._1(i)(j,k)+"\t")
        }
        print("\n")
      }
    }
/*
convolved_input_0:
-1.0	-8.0	-7.0	-7.0	-8.0	
-10.0	-14.0	-12.0	-12.0	-3.0	
-5.0	-9.0	-10.0	-11.0	-10.0	
-1.0	-11.0	-5.0	-5.0	-6.0	
-1.0	-2.0	-5.0	-3.0	0.0	
convolved_input_1:
0.0	-6.0	-4.0	-8.0	-8.0	
-2.0	0.0	2.0	0.0	-6.0	
-2.0	-5.0	-3.0	-5.0	-3.0	
2.0	-1.0	4.0	2.0	-3.0	
3.0	6.0	0.0	4.0	-1.0	
 * */  
    print("tesp2:test for Max_PoolLayer forward:\n")
    for(i<-0 until test_result1._2.length){
      print("pooled_input_"+i+":\n")
      for(j<-0 until test_result1._2(i).rows){
        for(k<-0 until test_result1._2(i).cols){
          print(test_result1._2(i)(j,k)+"\t")
        }
        print("\n")
      }
    }
/*
pooled_input_0:
0.2689414213699951	9.110511944006456E-4	
0.2689414213699951	0.006692850924284857	
pooled_input_1:
0.5	0.8807970779778823	
0.8807970779778823	0.9820137900379085	
 * */
    for(i<-0 until test_result1._3.length){
      print("pooled_input_maxvalue_zero_"+i+":\n")
      for(j<-0 until test_result1._3(i).rows){
        for(k<-0 until test_result1._3(i).cols){
          print(test_result1._3(i)(j,k)+"\t")
        }
        print("\n")
      }
    }
/*
pooled_input_maxvalue_zero_0:
0.2689414213699951	0.0	9.110511944006456E-4	0.0	0.0	
0.0	0.0	0.0	0.0	0.0	
0.0	0.0	0.0	0.0	0.0	
0.2689414213699951	0.0	0.006692850924284857	0.0	0.0	
0.0	0.0	0.0	0.0	0.0	
pooled_input_maxvalue_zero_1:
0.5	0.0	0.0	0.0	0.0	
0.0	0.0	0.8807970779778823	0.0	0.0	
0.0	0.0	0.0	0.0	0.0	
0.8807970779778823	0.0	0.9820137900379085	0.0	0.0	
0.0	0.0	0.0	0.0	0.0	
 * */
  }  
}
