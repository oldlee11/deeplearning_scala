package dp_process

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
 * 卷积层
 * input_size   输入数据的width,height
 *              如果是最底层则=处理图片的width,height(imgae_size) 
 *              如果是中间层则=是输入数据尺度
 * n_kernel     卷积核个数
 * kernel_size  卷积核的width,height
 * n_channel      图片组成的个数,一般一个图片是以RGB形式给出的,所以一般=3
 * rng
 * activation   卷积后经过的非线性处理函数
 */
class ConvLayer(input_size_in:(Int,Int),
                n_kernel_in:Int,
                kernel_size_in:(Int,Int),
                init_a:Double=0.0,
                _W:Array[Array[Array[Array[Double]]]]=null,
                _b:Array[Double]=null,
                n_channel_in:Int=3,
                _rng: Random=null,
                activation:String="ReLU") {
  
  var rng:Random=if(_rng == null) new Random(1234) else _rng
  
  /*
   * 其它全局变量
   * */  
  val input_size:(Int,Int)=input_size_in
  val n_kernel:Int=n_kernel_in
  val kernel_size:(Int,Int)=kernel_size_in
  val n_channel:Int=n_channel_in
  //一个卷积核对一个image的一个channel数据经过卷积后输出的矩阵的宽和高=s0*s1
  val s0:Int = input_size._1 - kernel_size._1 + 1
  val s1:Int = input_size._2 - kernel_size._2 + 1  
  //print(input_size._1+"\t"+kernel_size._1+"\t"+s0+"\n")//debug
  //print(input_size._1+"\t"+kernel_size._2+"\t"+s1+"\n")//debug
  //用于正向传播,activated_input也用于反向传播
  var convolved_input:Array[Array[Array[Double]]]=Array.ofDim[Double](n_kernel,s0,s1)//convolved_input=一个channel数据,在经过卷积网络处理后(n_kernel个核) 输出数据的size=n_kernel个 s0*s1
  var activated_input:Array[Array[Array[Double]]]=Array.ofDim[Double](n_kernel,s0,s1)//activated_input=convolved_input经过activation_fun函数处理的输出  
  //用于反向传播
  var d_v:Array[Array[Array[Double]]]=Array.ofDim[Double](n_kernel,s0,s1)//网络的局部梯度  
  var W_add:Array[Array[Array[Array[Double]]]]=Array.ofDim[Double](n_kernel,n_channel,kernel_size._1, kernel_size._2)//W的更新大小
  var b_add:Array[Double]=new Array[Double](n_kernel)//b的更新大小    
  
  /*
   * 建立参数W和b
   */
  var W:Array[Array[Array[Array[Double]]]]=Array.ofDim[Double](n_kernel,n_channel,kernel_size._1, kernel_size._2)
  var b:Array[Double]=new Array[Double](n_kernel)
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
   * 使用一个样本做forward向前处理
   * x:一个输入样本的x值=>n_channel*input_size._1*input_size._2 三维给出
   * 详细原理参见[参考资料>CNN>CNN_forward.jpg]
   * 输出=修改activated_input
   * 相当于 convolved_input=convn(x,rot180(W),'vaild')------二维相关操作 convn是matlab的卷积操作,rot180表示反转180度
   * */
  def convolve_forward(x: Array[Array[Array[Double]]]){
    //最后输出每个核对图片数据的卷积处理后的数据=n_kernel*s0*s1
    //遍历每个核(使用某一个核来卷积处理图像数据)
    for(k <- 0 until n_kernel){
      //遍历经过某一个核卷积后输出样本矩阵的长度
      for(i <- 0 until s0){ 
        //遍历经过某一个核卷积后输出样本矩阵的宽度
        for(j <- 0 until s1){
          convolved_input(k)(i)(j)=0.0//情空
          //遍历每个channel(使用某个核来卷积处理每个channel数据)
          for(c <- 0 until n_channel){
            //遍历核内的长度
            for(s <- 0 until kernel_size._1){
              //遍历核内的高度
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
  }

  
  /* 下一层是poollayer层
   * 使用一个样本做backward向后处理
   * x:一个输入样本的x值=>n_channel*input_size._1*input_size._2 三维给出
   * next_layer:下一层网络
   * lr:学习率
   * batch_num 是一次批量迭代使用的样本数,由于每一次训练时,样本仅输入1个,参数(W,vbias,hbias)迭代计算时：W(t+1)=W(t)+lr*(局部误差/batch_num)
                                                 即每一次迭代仅仅更新1/batch_num倍,如果batch_num=训练集样本个数,则当所有训练集都迭代一遍后则 为 W(t+1)=W(t)+lr*(局部误差)
                                                如果batch_num=1则表示每个样本迭代一次 W(t+1)=W(t)+lr*(局部误差) 即不是批量迭代而是随机迭代了  
   * */
  def convolve_backward(x: Array[Array[Array[Double]]],
                          next_layer:Max_PoolLayer, 
                          lr: Double,
                          batch_num:Int,alpha:Double=0.0){
    /*
     * 计算局部梯度
     */
    var tmp1:Int=0
    var tmp2:Int=0  
    d_v=Array.ofDim[Double](n_kernel,s0,s1)//清空上次的d_v
    //遍历下一个max池化层每个核(使用某一个核来卷积处理图像数据)
    for(k <- 0 until next_layer.pre_conv_layer_n_kernel){
      //遍历经过某一个池化层后输出样本矩阵的长度
      for(i <- 0 until next_layer.s0){ 
        //遍历经过某一个池化层后输出样本矩阵的高度
        for(j <- 0 until next_layer.s1){
          //参考 《CNN的反向求导及联系.pdf》中的问题2
          d_v(k)(next_layer.max_index_x(k)(i)(j)._1)(next_layer.max_index_x(k)(i)(j)._2)=next_layer.d_v(k)(i)(j)*dactivation_fun(convolved_input(k)(next_layer.max_index_x(k)(i)(j)._1)(next_layer.max_index_x(k)(i)(j)._2))          
        }
      }
    }

    /*
     * 计算参数的更新大小    
     * */
    var W_add_tmp:Array[Array[Array[Array[Double]]]]=Array.ofDim[Double](n_kernel,n_channel,kernel_size._1, kernel_size._2)//W的更新大小
    var b_add_tmp:Array[Double]=new Array[Double](n_kernel)//b的更新大小    
    //遍历每个核(使用某一个核来卷积处理图像数据)
    for(k <- 0 until n_kernel){
      //遍历经过某一个核卷积后输出样本矩阵的长度
      for(i <- 0 until s0){ 
        //遍历经过某一个核卷积后输出样本矩阵的宽度
        for(j <- 0 until s1){
          b_add_tmp(k) +=d_v(k)(i)(j)*1.0
          for(c <- 0 until n_channel){
            //遍历核内的长度
            for(s <- 0 until kernel_size._1){
              //遍历核内的高度
              for(t <- 0 until kernel_size._2){
                //应该实现matlab的 W_add_tmp=convn(x,rot180(d_v),'valid'),//实际就是卷积层的正向传播,参考convolve_forward的实现
                W_add_tmp(k)(c)(s)(t) += x(c)(i+s)(j+t)*d_v(k)(i)(j) 
              }
            }
          }          
        }
      }
    }
    /* debug
    x.foreach { x => print("x:\n");x.foreach { x => x.foreach { x => print(x+"\t") };print("\n") } }//debug
    d_v.foreach { x => print("d_v:\n");x.foreach { x => x.foreach { x => print(x+"\t") };print("\n") } }//debug
    print("W_add_0_0:\n")//debug
    W_add_tmp(0)(0).foreach { x => x.foreach { x => print(x+"\t") };print("\n") }//debug
    */
    /*
     * 更新参数
     * */
    for(k <- 0 until n_kernel){
      //b(k) -= lr * b_add(k) / batch_num  //yusugomori?????????????????????
      //b_add(k) =lr *b_add_tmp(k)/batch_num//vision1
      b_add(k) =alpha*b_add(k)+ lr *b_add_tmp(k)/batch_num//vision2
      b(k) += b_add(k) 
      //遍历每个channel(使用某个核来卷积处理每个channel数据)
      for(c <- 0 until n_channel){
        //遍历核内的长度
        for(s <- 0 until kernel_size._1){
          //遍历核内的高度
          for(t <- 0 until kernel_size._2){  
            //W(k)(c)(s)(t) -= lr * W_add(k)(c)(s)(t) / batch_num    //yusugomori?????????????
            //W_add(k)(c)(s)(t)=lr*W_add_tmp(k)(c)(s)(t)/batch_num //vision1
            W_add(k)(c)(s)(t)=alpha*W_add(k)(c)(s)(t)+lr*W_add_tmp(k)(c)(s)(t)/batch_num//vision2
            W(k)(c)(s)(t) += W_add(k)(c)(s)(t)                       //使用加法是由于 logisticregression和hidden的都是加法,
                                                                     //本质是输出层logisticregression的d_y(i) = y(i) - p_y_given_x_softmax(i)  如果是p_y_given_x_softmax(i)-y(i) 则统一为减法
          }
        }
      }  
    }
  }  
  
  
  /*
   * 函数
   */  
    def uniform(min: Double, max: Double): Double = rng.nextDouble() * (max - min) + min
    def activation_fun(linear_output:Double):Double={
      if(activation=="sigmoid"){
        utils.sigmoid(linear_output)  
      }else if(activation=="tanh"){
        utils.tanh(linear_output)
      }else if(activation=="ReLU") {
        utils.ReLU(linear_output)
      }else if(activation=="linear"){
        linear_output
      }else{
        0.0
      }
    }
    def dactivation_fun(linear_output:Double):Double={
      if(activation=="sigmoid"){
        utils.dsigmoid(linear_output)  
      }else if(activation=="tanh"){
        utils.dtanh(linear_output)
      }else if(activation=="ReLU") {
        utils.dReLU(linear_output)
      }else if(activation=="linear"){
        1.0
      }else{
        0.0
      }
    }  

  //记录系数
  def save_w(file_module_in:String):Unit={
   val writer_module = new PrintWriter(new File(file_module_in)) 
   //write b
   writer_module.write(b.mkString(sep=","))  
   writer_module.write("\n")
   //write w
   for(i<-0 until W.length){
     for(j<-0 until W(0).length){
       for(k<-0 until W(0)(0).length){
         writer_module.write(W(i)(j)(k).mkString(sep=","))
         writer_module.write("\n")         
       }
     }
   }
   writer_module.close()
  }
  //读取 save_w输出的模型txt
  //并修改dA的W    vbias    hbias
  def read_w_module(file_module_in:String):Unit={
    var result:ArrayBuffer[String]=ArrayBuffer();
    var tmp:Array[Double]=Array()
    Source.fromFile(file_module_in).getLines().foreach(line=>result+=line) 
    b=result(0).split(",").map(x=>x.toDouble)
    for(i<-0 until W.length){
      for(j<-0 until W(0).length){
        for(k <- 0 until W(0)(0).length){
          tmp=result(1+i*W(0).length*W(0)(0).length+j*W(0)(0).length).split(",").map(x=>x.toDouble) 
          for(m<-0 until W(0)(0)(0).length){
            W(i)(j)(k)(m)=tmp(m)  
          }
        }
      }
    }
  }    
    
}

//池化层
/*
 * input_size:上一层卷积层的输出的长和高
 * pre_conv_layer_n_kernel:=上一层卷积层的n_kernel
 * pool_size:=赤化的长和高,把input_size按照pool_size大小平分,并对每一份取max,最后把输入数据做sampleing,这里没有参数W和b
 * */
class Max_PoolLayer(input_size:(Int,Int),
                    pre_conv_layer_n_kernel_in:Int,
                    pool_size_in:(Int,Int),
                    _rng: Random=null) {
  
  var rng:Random=if(_rng == null) new Random(1234) else _rng 

  
  /*
   * 其它全局变量
   * */  
  val pre_conv_layer_n_kernel:Int=pre_conv_layer_n_kernel_in
  val pool_size:(Int,Int)=pool_size_in
  //一个卷积核对一个image的一个channel数据经过卷积后输出的矩阵的宽和高=s0*s1
  val s0_double:Double=(input_size._1.toDouble)/(pool_size._1.toDouble);
  val s1_double:Double=(input_size._2.toDouble)/(pool_size._2.toDouble);
  /*如果图片尺寸和卷积核尺寸不是整除,则很可能有向量角标异常,所以使用向下取证,牺牲部分边缘图像不处理
  val s0:Int = math.ceil(s0_double).toInt//向上取整
  val s1:Int = math.ceil(s1_double).toInt//向上取整 
  val s0:Int = math.round(s0_double).toInt//四舍五入取整
  val s1:Int = math.round(s1_double).toInt//四舍五入取整   
  */
  val s0:Int = math.floor(s0_double).toInt//向下取整
  val s1:Int = math.floor(s1_double).toInt//向下取整 
  
  val max_index_x:Array[Array[Array[(Int,Int)]]]=Array.ofDim[(Int,Int)](pre_conv_layer_n_kernel,s0,s1)//用于记录输出的max 对应输入的坐标
  
  //用于正向传播
  var pooled_input:Array[Array[Array[Double]]]=Array.ofDim[Double](pre_conv_layer_n_kernel,s0,s1)//把输入经过max池化处理的数据输出
  //用于反向传播
  var d_v:Array[Array[Array[Double]]]=Array.ofDim[Double](pre_conv_layer_n_kernel,s0,s1)//网络的局部梯度    
  
  /*
   * 使用一个样本做forward向前处理
   * x:一个输入样本的x值=>上一层卷积层经过正向处理(convolve_forward方法)的输出activated_input
   * 输出=修改pooled_input
   * */
  def maxpoollayer_forward(x: Array[Array[Array[Double]]]){
    var max_index_1:Int=0;
    var max_index_2:Int=0;
    var max_tmp:Double=0.0;
    var x_index_1:Int=0;
    var x_index_2:Int=0;
    //最后输出每个max池化核对数据处理后的数据=pre_conv_layer_n_kernel*s0*s1
    //遍历每个核(使用某一个核来卷积处理图像数据)
    for(k <- 0 until pre_conv_layer_n_kernel){      
      //遍历经过某一个核卷积后输出样本矩阵的长度
      for(i <- 0 until s0){ 
        //遍历经过某一个核卷积后输出样本矩阵的宽度
        for(j <- 0 until s1){          
          //遍历max池化核内的长度         
          for(s <- 0 until pool_size._1){
            //遍历max池化核内的高度
            for(t <- 0 until pool_size._2){    
              x_index_1=pool_size._1*i+s
              x_index_2=pool_size._2*j+t
              if(s==0 && t==0){
                max_tmp=x(k)(x_index_1)(x_index_2)
                max_index_1=x_index_1
                max_index_2=x_index_2
              }
              if(max_tmp<x(k)(x_index_1)(x_index_2)){
                max_tmp=x(k)(x_index_1)(x_index_2)//取pool_size._1 * pool_size._2内的最大值
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
  }  

  /* 下一层是convlayer卷积层
   * 使用一个样本做backward向后处理
   * x:一个输入样本的x值
   * next_layer:下一层网络
   * lr:学习率
   * batch_num 是一次批量迭代使用的样本数,由于每一次训练时,样本仅输入1个,参数(W,vbias,hbias)迭代计算时：W(t+1)=W(t)+lr*(局部误差/batch_num)
                                                 即每一次迭代仅仅更新1/batch_num倍,如果batch_num=训练集样本个数,则当所有训练集都迭代一遍后则 为 W(t+1)=W(t)+lr*(局部误差)
                                                如果batch_num=1则表示每个样本迭代一次 W(t+1)=W(t)+lr*(局部误差) 即不是批量迭代而是随机迭代了  
   * */
  
  //方案二   参考《CNN的反向求导及联系中的问题三》
  //核心代码参考 【反向传播 卷积计算.jpg】
  def maxpoollayer_backward_1(x: Array[Array[Array[Double]]],
                              next_layer:ConvLayer){ 
    d_v=Array.ofDim[Double](pre_conv_layer_n_kernel,s0,s1)//清空为0.0
    var tmp1:Int=0
    var tmp2:Int=0    
    for (c<- 0 until next_layer.n_channel){
      for(tmp1<- 0 until next_layer.d_v(0).length){
        for(tmp2<- 0 until next_layer.d_v(0)(0).length){
          for(k <- 0 until next_layer.n_kernel){
            for(s <- 0 until next_layer.kernel_size._1){
              for(t <- 0 until next_layer.kernel_size._2){
                //相当于MATLAB中的convn(next_layer.d_v,next_layer.W,'full'),即完成了2维的卷积运算
                d_v(c)(tmp1+s)(tmp2+t) += next_layer.d_v(k)(tmp1)(tmp2) * next_layer.W(k)(c)(s)(t)
                //由于是max 所以没有 *dactivation_fun(input) 
                
              }
            }
          }  
        }
      }
    }
    /*
     * 计算参数的更新大小    W_add  b_add
     * 由于是max操作,没有参数W和b,所以没有W_add  b_add 也不用更新
     * */     
  }
  
  
  /* 下一层是HiddenLayer输出层  由于是全连接,所以就是简单的,注意mapxpoollayer的d_v要以一维形式来处理
   * 使用一个样本做backward向后处理
   * x:一个输入样本的x值
   * next_layer:下一层网络
   * */
  def maxpoollayer_backward_2(x: Array[Array[Array[Double]]],
                              next_layer:Dropout){ 
    
    /*
     * 计算局部梯度
     */
    /*
    //遍历 mapxpoollayer的输出 也就是遍历next_layer.n_in
    d_v=Array.ofDim[Double](pre_conv_layer_n_kernel,s0,s1)//清空为0.0
    for(k <- 0 until pre_conv_layer_n_kernel){
      for(i <- 0 until s0){ 
        for(j <- 0 until s1){  
          //遍历next_layer.n_out
          for(g <-0 until next_layer.n_out){
            d_v(k)(i)(j)= d_v(k)(i)(j) + next_layer.W(g)(k*s0*s1+i*s1+j) * next_layer.d_v(g)  
          } 
          //由于是max 所以没有 *dactivation_fun(input)
        }
      }
    }
    */
    //使用下一层的d_v累加计算出本层的d_v,但是是打平后的
    val flatten_size:Int=next_layer.n_in
    val d_v_flatten:Array[Double]=new Array(flatten_size)//内部为0.0 每次清零
    val last_hidden_index:Int=next_layer.hidden_layers.length-1
    for(j <-0 until flatten_size){
      for(i <-0 until next_layer.hidden_layers(last_hidden_index).n_out){
        d_v_flatten(j) = d_v_flatten(j) + next_layer.hidden_layers(last_hidden_index).W(i)(j) * next_layer.hidden_layers(last_hidden_index).d_v(i)
      }
    }
    //把打平后的d_v变为k,i,j
    var index:Int=0
    for(k <- 0 until pre_conv_layer_n_kernel){
      for(i <- 0 until s0){ 
        for(j <- 0 until s1){ 
          d_v(k)(i)(j)=d_v_flatten(index)//由于是max 所以没有 *dactivation_fun(input)
          index +=1
        }
      }
    }   
  }
  
}

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
class ConvPoolLayer(input_size_in:(Int,Int),
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
  val init_a_tmp:Double=math.sqrt(6.0/(f_in_tmp + f_out_tmp)) 
  //val init_a_tmp:Double=1/ math.pow(f_out_tmp,0.25)  //cnn simple
  val ConvLayer_obj:ConvLayer= new ConvLayer(input_size_in=input_size_in,
                                             n_kernel_in=n_kernel_in,
                                             kernel_size_in=kernel_size_in,
                                             _W=_W,
                                             _b=_b,
                                             init_a=init_a_tmp,
                                             n_channel_in=n_channel_in,
                                             _rng=rng,
                                             activation=activation)
  val Max_PoolLayer_obj:Max_PoolLayer=new Max_PoolLayer(input_size=(ConvLayer_obj.s0,ConvLayer_obj.s1),
                                                        pre_conv_layer_n_kernel_in=ConvLayer_obj.n_kernel,
                                                        pool_size_in=pool_size_in,
                                                        _rng=rng)
  
  def output(x: Array[Array[Array[Double]]]):Array[Array[Array[Double]]]={
    ConvLayer_obj.convolve_forward(x=x)
    Max_PoolLayer_obj.maxpoollayer_forward(x=ConvLayer_obj.activated_input)
    Max_PoolLayer_obj.pooled_input
  }

  def cnn_forward(x:Array[Array[Array[Double]]])={
    ConvLayer_obj.convolve_forward(x)
    Max_PoolLayer_obj.maxpoollayer_forward(x=ConvLayer_obj.activated_input)
  }
  
  def cnn_backward_1(x:Array[Array[Array[Double]]],next_layer:ConvPoolLayer, lr:Double,batch_num:Int,alpha:Double=0.0)={
    Max_PoolLayer_obj.maxpoollayer_backward_1(x=ConvLayer_obj.activated_input,next_layer=next_layer.ConvLayer_obj)
    ConvLayer_obj.convolve_backward(x=x, next_layer=Max_PoolLayer_obj, lr=lr, batch_num=batch_num,alpha=alpha)  
  }
  
  def cnn_backward_2(x:Array[Array[Array[Double]]],next_layer:Dropout, lr:Double,batch_num:Int,alpha:Double=0.0)={
    Max_PoolLayer_obj.maxpoollayer_backward_2(x=ConvLayer_obj.activated_input,next_layer=next_layer)
    ConvLayer_obj.convolve_backward(x=x, next_layer=Max_PoolLayer_obj, lr=lr, batch_num=batch_num,alpha=alpha)  
  }
  
}

object ConvPoolLayer{
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
    var ConvPoolLayer_obj_test:ConvPoolLayer =new ConvPoolLayer(input_size_in=(7,7),
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
    ConvPoolLayer_obj_test.cnn_forward(x=x_test)
    ConvPoolLayer_obj_test.ConvLayer_obj.convolved_input.foreach { x => print("convolved_input_i:\n");x.foreach { x => x.foreach { x => print(x+"\t") } ;print("\n")}}
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
    ConvPoolLayer_obj_test.ConvLayer_obj.activated_input.foreach { x => print("activated_input_i:\n");x.foreach { x => x.foreach { x => print(x+"\t") } ;print("\n")}}
/*
activated_input_i:
0.2689414213699951	  3.3535013046647827E-4	9.110511944006456E-4	9.110511944006456E-4	3.3535013046647827E-4	
4.539786870243442E-5	8.315280276641327E-7	6.1441746022147215E-6	6.1441746022147215E-6	0.04742587317756679	
0.006692850924284857	1.233945759862318E-4	4.539786870243442E-5	1.670142184809519E-5	4.539786870243442E-5	
0.2689414213699951	  1.670142184809519E-5	0.006692850924284857	0.006692850924284857	0.002472623156634775	
0.2689414213699951	  0.11920292202211757	  0.006692850924284857	0.04742587317756679	  0.5	
activated_input_i:
0.5	                  0.002472623156634775	  0.017986209962091562	3.3535013046647827E-4	3.3535013046647827E-4	
0.11920292202211757	  0.5	                    0.8807970779778823	  0.5	                  0.002472623156634775	
0.11920292202211757	  0.006692850924284857	  0.04742587317756679	  0.006692850924284857	0.04742587317756679	
0.8807970779778823	  0.2689414213699951	    0.9820137900379085	  0.8807970779778823	  0.04742587317756679	
0.9525741268224331	  0.9975273768433653	    0.5	                  0.9820137900379085	  0.2689414213699951
 * */    
    ConvPoolLayer_obj_test.Max_PoolLayer_obj.maxpoollayer_forward(x=ConvPoolLayer_obj_test.ConvLayer_obj.activated_input)
    print("tesp2:test for Max_PoolLayer forward:\n")
    ConvPoolLayer_obj_test.Max_PoolLayer_obj.pooled_input.foreach { x => print("pooled_input_i:\n");x.foreach { x => x.foreach { x => print(x+"\t") } ;print("\n")}}
/*
pooled_input_i:
0.2689414213699951	9.110511944006456E-4	
0.2689414213699951	0.006692850924284857	
pooled_input_i:
0.5	                0.8807970779778823	
0.8807970779778823	0.9820137900379085
 * */
    ConvPoolLayer_obj_test.Max_PoolLayer_obj.max_index_x.foreach { x => print("max_index_x_i:\n");x.foreach { x => x.foreach { x => print("("+x._1+","+x._2+")\t") } ;print("\n")}}
/*
 max_index_x_i:
(0,0)	(0,2)	
(3,0)	(3,2)	
max_index_x_i:
(0,0)	(1,2)	
(3,0)	(3,2)	
 */    
    
    //ok
    ////数据案例使用 《CNN的反向求导及联系.pdf》中的问题三
    print("step3: test for Max_PoolLayer backward_1(maxpool的下一层是卷积层conv) :\n")
    var init_w_2:Array[Array[Array[Array[Double]]]]=Array(
        Array(
            Array(
                Array(0.1,0.2), 
                Array(0.2,0.4)
            )            
        ),
        Array(
            Array(
                Array(-0.3,0.1), 
                Array(0.1,0.2)
            )             
        )
    )    
    var ConvPoolLayer_obj_test_2_next:ConvPoolLayer =new ConvPoolLayer(input_size_in=(3,3),
                                                                n_kernel_in=2,
                                                                kernel_size_in=(2,2),
                                                                pool_size_in=(1,1),
                                                                _W=init_w_2,
                                                                n_channel_in=1,
                                                                activation="sigmoid")
    var init_d_v_2:Array[Array[Array[Double]]]=Array(
        Array(
           Array(1,3),
           Array(2,2)
        ),
        Array(            
           Array(2,1),
           Array(1,1)
        )
    )
       
    ConvPoolLayer_obj_test_2_next.ConvLayer_obj.d_v=init_d_v_2
    var ConvPoolLayer_obj_test_2:ConvPoolLayer =new ConvPoolLayer(input_size_in=(10,10),
                                                                n_kernel_in=1,
                                                                kernel_size_in=(5,5),
                                                                pool_size_in=(2,2),
                                                                n_channel_in=1,
                                                                activation="sigmoid") 
    ConvPoolLayer_obj_test_2.Max_PoolLayer_obj.maxpoollayer_backward_1(x=Array(Array(Array(1.0,0.1,0.11,0.34,0.91,0.01),
                                                                                     Array(1.0,0.1,0.11,0.34,0.91,0.01),
                                                                                     Array(1.0,0.1,0.11,0.34,0.91,0.01),
                                                                                     Array(1.0,0.1,0.11,0.34,0.91,0.01),
                                                                                     Array(1.0,0.1,0.11,0.34,0.91,0.01),
                                                                                     Array(1.0,0.1,0.11,0.34,0.91,0.01))), 
                                                                       next_layer=ConvPoolLayer_obj_test_2_next.ConvLayer_obj)
    ConvPoolLayer_obj_test_2.Max_PoolLayer_obj.d_v.foreach { x => print("d_v_i:\n");x.foreach { x => x.foreach { x => print(x+"\t") };print("\n") } }
/*
d_v_i:
-0.5	0.4000000000000001	0.7000000000000001	
0.3000000000000001	1.9000000000000006	1.9000000000000004	
0.5	1.5	1.0	
d_v_i=conv(nextlayer.d_v,nextlayer.w,'full')
 * */  
    
    //ok
    //数据案例使用 《CNN的反向求导及联系.pdf》中的问题2
    print("step4: test for ConvLayer backward:\n") 
    var ConvPoolLayer_obj_test3:ConvPoolLayer =new ConvPoolLayer(input_size_in=(7,7),
                                                                 n_kernel_in=1,
                                                                 kernel_size_in=(4,4),
                                                                 pool_size_in=(2,2),
                                                                 n_channel_in=1,
                                                                 activation="sigmoid")    
    ConvPoolLayer_obj_test3.ConvLayer_obj.convolve_forward(x=Array(Array(Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0))))
    ConvPoolLayer_obj_test3.Max_PoolLayer_obj.maxpoollayer_forward(ConvPoolLayer_obj_test3.ConvLayer_obj.activated_input)
    ConvPoolLayer_obj_test3.Max_PoolLayer_obj.max_index_x(0)(0)(0)=(1,1)
    ConvPoolLayer_obj_test3.Max_PoolLayer_obj.max_index_x(0)(0)(1)=(0,3)
    ConvPoolLayer_obj_test3.Max_PoolLayer_obj.max_index_x(0)(1)(0)=(2,0)
    ConvPoolLayer_obj_test3.Max_PoolLayer_obj.max_index_x(0)(1)(1)=(3,2)
    ConvPoolLayer_obj_test3.Max_PoolLayer_obj.d_v=Array(Array(Array(1,3),Array(2,4)))
    ConvPoolLayer_obj_test3.ConvLayer_obj.convolve_backward(x=Array(Array(Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0),
                                                                         Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0))),
                                                             next_layer=ConvPoolLayer_obj_test3.Max_PoolLayer_obj, lr=0.1, batch_num=1,alpha=0.0)
    //打开 ConvPoolLayer_obj_test3.ConvLayer_obj.convolve_backward 
    //打开x.foreach { x => print("x:\n");x.foreach { x => x.foreach { x => print(x+"\t") };print("\n") } }//debug
    //d_v.foreach { x => print("d_v:\n");x.foreach { x => x.foreach { x => print(x+"\t") };print("\n") } }//debug
    //print("W_add_0_0:\n")//debug
    //W_add(0)(0).foreach { x => x.foreach { x => print(x+"\t") };print("\n") }//debug
    /*
x:
1.0	2.0	3.0	4.0	5.0	6.0	7.0	
1.0	2.0	3.0	4.0	5.0	6.0	7.0	
1.0	2.0	3.0	4.0	5.0	6.0	7.0	
1.0	2.0	3.0	4.0	5.0	6.0	7.0	
1.0	2.0	3.0	4.0	5.0	6.0	7.0	
1.0	2.0	3.0	4.0	5.0	6.0	7.0	
1.0	2.0	3.0	4.0	5.0	6.0	7.0	
d_v:
0.0	0.0	0.0	-168.04927028385532	
0.0	-26.265052184478442	0.0	0.0	
-31.075304665281394	0.0	0.0	0.0	
0.0	0.0	-159.03190353166173	0.0	
W_add_0_0:
-1232.8982007646448	-1617.3197314299218	-2001.7412620951986	-2386.1627927604754	
-1232.8982007646448	-1617.3197314299218	-2001.7412620951986	-2386.1627927604754	
-1232.8982007646448	-1617.3197314299218	-2001.7412620951986	-2386.1627927604754	
-1232.8982007646448	-1617.3197314299218	-2001.7412620951986	-2386.1627927604754
W_add_0_0=rot180(convn(x,rot180(d_v)),'valid'))
     * */    
    
  }   
  
}