package dp_process

import scala.util.Random
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array
import scala.math

import java.io.File; 
import java.io._
import scala.io.Source


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
                var rng: Random=null,
                activation:String="ReLU") {
  
  if(rng == null) rng = new Random(1234) 
  
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
                //做卷积运算  核内的元素和输入样本x的元素对应xiangcheng后计算求和
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
                //应该实现matlab的 rot180(convn(x,rot180(d_v),'valid')),
                //d_v(k)(i)(j) * x(c)(i+s)(j+t)完成了convn(x,d_v,'valid')//实际就是卷积层的正向传播,参考convolve_forward的实现
                //其中再把d_v(k)(i)(j)->d_v(k)((s0-1)-i)((s1-1)-j)实现了rot180(d_v)
                //输出W_add(k)(c)(s)(t)->W_add(k)(c)((kernel_size._1-1)-s)((kernel_size._2-1)-t) 事先最后的rot180
                W_add_tmp(k)(c)((kernel_size._1-1)-s)((kernel_size._2-1)-t) += x(c)(i+s)(j+t)*d_v(k)((s0-1)-i)((s1-1)-j) 
                //W_add_tmp(k)(c)((kernel_size._1-1)-s)((kernel_size._2-1)-t) += x(c)(i+s)(j+t)*d_v(k)(i)(j) 
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
    
    
    
  /*
   * 用于 pretrain 预处理(da方法)
   * 模拟 dA
   * */
  var cross_entropy_result:Double=0.0
  var vbias:Array[Double]=new Array[Double](n_channel)
  //使用交叉信息嫡cross-entropy衡量样本输入和经过编解码后输出的相近程度
  //值在>=0 当为0时 表示距离接近
  //每次批量迭代后该值越来越小
  def cross_entropy(x: Array[Array[Array[Double]]], z:  Array[Array[Array[Double]]]):Double={
    var result:Double=0.0
    for(i <-0 until x.length){
      for(j <-0 until x(0).length){
        for(k <-0 until x(0)(0).length){
          result += x(i)(j)(k)*math.log(z(i)(j)(k))+(1-x(i)(j)(k))*math.log(1-z(i)(j)(k))
        }
      }  
    }
   -1.0 * result
  }  
  //定义二项分布的随机数产生函数
  //n是二项分布数据的个数,例如=100个
  //p是二项分=1的概率值,例如=0.2
  //返回值=n中取值=1的个数约等于 =100*0.2=20 ,随机值
  //如果n=1则为产生一个二项分布的随机数
  def binomial(n: Int, p: Double): Double = {
      if(p < 0 || p > 1) return 0
      var c: Int = 0
      var r: Double = 0
      var i: Int = 0
      for(i <- 0 until n) {
        r = rng.nextDouble()
        if(r < p) c += 1
      }
      c.toDouble
  }    
  //把一维向量x对根据二项分布p 加入噪音,最后生成tilde_x
  //如果原始数据=0,则依旧=0
  //如果原始数据不等于0,则=binomial(1, p)   取值为0或1的值,随机生成1的概率=p
  def get_corrupted_input(x: Array[Array[Array[Double]]], tilde_x:Array[Array[Array[Double]]], p: Double) {
      for(i <-0 until x.length){
        for(j<-0 until x(0).length){
          for(k<-0 until x(0)(0).length){
            if(x(i)(j)(k) == 0) {
              tilde_x(i)(j)(k) = 0.0;
            } else {
              tilde_x(i)(j)(k) = binomial(1, p)*x(i)(j)(k)
            }            
          }
        }
      }
  }
  // Encode 编码过程：v-->h  代码几乎等于向前传播convolve_forward
  // 把可视层(v)内的单元的值 经过v->h连接的系数W加权求和并加上隐藏层单元的偏置,最后经过sigmoid映射为[0,1]数值=隐藏层(h)内每个单元的数值=y
  def get_hidden_values(x: Array[Array[Array[Double]]], y: Array[Array[Array[Double]]]) {
    val tmp:Array[Array[Array[Double]]]=Array.ofDim[Double](y.length,y(0).length,y(0)(0).length)
    for(k <- 0 until n_kernel){
      for(i <- 0 until s0){ 
        for(j <- 0 until s1){
          for(c <- 0 until n_channel){
            for(s <- 0 until kernel_size._1){
              for(t <- 0 until kernel_size._2){
                tmp(k)(i)(j) += W(k)(c)(s)(t) * x(c)(i+s)(j+t)
              }
            }
          }
          tmp(k)(i)(j)=tmp(k)(i)(j) + b(k)
          y(k)(i)(j) = utils.sigmoid(tmp(k)(i)(j))
        }
      }
    } 
  }
  // Decode  解码过程由:h-->v
  // 把隐藏层(h)内的单元的值 经过h->v连接的系数W加权求和并加上可视层单元的偏置,最后经过sigmoid映射为[0,1]数值=可视层(v)内每个单元的数值=z
  def get_reconstructed_input(y: Array[Array[Array[Double]]], z:Array[Array[Array[Double]]]) {
    val tmp:Array[Array[Array[Double]]]=Array.ofDim[Double](n_channel,input_size._1,input_size._2)
    //计算反向求和
    for(c <- 0 until n_channel){
      //遍历y
      for(k <- 0 until n_kernel){
        for(i <- 0 until s0){ 
          for(j <- 0 until s1){
            //遍历k内核
            for(s <- 0 until kernel_size._1){
              for(t <- 0 until kernel_size._2){
                tmp(c)(s+i)(t+j) += y(k)(i)(j) * W(k)(c)(s)(t)
              }
            }
          }
        }
      }
    }  
    //添加偏差量  并经过sigmoid处理
    for(c <- 0 until n_channel){
      for(i <-0 until input_size._1){
        for(j<-0 until input_size._2){
          z(c)(i)(j)=tmp(c)(i)(j)+vbias(c)
          z(c)(i)(j)=utils.sigmoid(z(c)(i)(j))
        }
      }
    }
  } 
  //pretrain 预处理  da 方式
  def pre_train_da(x: Array[Array[Array[Double]]], lr: Double, corruption_level: Double,batch_num:Int){
      var tilde_x: Array[Array[Array[Double]]] = Array.ofDim[Double](x.length,x(0).length,x(0)(0).length)//经过加噪音处理后的x输入
      var y: Array[Array[Array[Double]]] = Array.ofDim[Double](n_kernel,s0,s1) //编码后的数据
      var z: Array[Array[Array[Double]]] = Array.ofDim[Double](x.length,x(0).length,x(0)(0).length) //解码后的数据     
      val p: Double = 1 - corruption_level
      //  x--->加噪音--->tilde_x
      get_corrupted_input(x, tilde_x, p)
      // tilde_x----->编码---->y---->解码---->z
      get_hidden_values(tilde_x,y)//完成一次编码,输出=y
      get_reconstructed_input(y, z)//完成一次解码,输出=z
      cross_entropy_result += cross_entropy(x,z)//衡量输入和经过编解码后的输出数据之间的相似度(使用KL散度,即相对信息嫡) 
      /* vbias迭代 */
      val L_vbias: Array[Array[Array[Double]]] = Array.ofDim[Double](x.length,x(0).length,x(0)(0).length)//x和z的误差
      for(i<- 0 until x.length){
        for(j<- 0 until x(0).length){
          for(k<- 0 until x(0)(0).length){
            L_vbias(i)(j)(k)=x(i)(j)(k)-z(i)(j)(k)
            vbias(i) += lr * L_vbias(i)(j)(k)/batch_num
          }
        }
      }
      /* b迭代 */
      val L_hbias:Array[Array[Array[Double]]] = Array.ofDim[Double](y.length,y(0).length,y(0)(0).length)
      for(k <- 0 until n_kernel){
        for(i <- 0 until s0){ 
          for(j <- 0 until s1){
            for(c <- 0 until n_channel){
              for(s <- 0 until kernel_size._1){
                for(t <- 0 until kernel_size._2){
                  L_hbias(k)(i)(j) += W(k)(c)(s)(t) * L_vbias(c)(i+s)(j+t)
                }
              }
            }
            L_hbias(k)(i)(j) *=utils.dsigmoid(y(k)(i)(j))
            b(k) += lr * L_hbias(k)(i)(j) / batch_num
          }
        }
      }      
      /* W迭代 */
      for(k <- 0 until n_kernel){
        for(i <- 0 until s0){ 
          for(j <- 0 until s1){
            for(c <- 0 until n_channel){
              for(s <- 0 until kernel_size._1){
                for(t <- 0 until kernel_size._2){
                  W(k)(c)(s)(t) += lr* (L_hbias(k)(i)(j)*tilde_x(c)(i+s)(j+t)+L_vbias(c)(i+s)(j+t)*y(k)(i)(j))/batch_num
                }
              }
            }
          }
        }
      }      
  }
  def pre_train_da_batch(inputs: Array[Array[Array[Array[Double]]]], lr: Double, corruption_level: Double,batch_num_per:Double):Unit={
    //抽取样本个数
    val batch_num:Int=if(batch_num_per==1.0){
      inputs.length
    }else{
      math.round((inputs.length*batch_num_per).toFloat)//每次批量训练样本数
    }
    cross_entropy_result=0.0//每次批量开始时,把上次的交叉信息嫡清零  
    
    //完成一次批量训练
    val rng_epooch:Random=new Random()//每次生成一个种子
    val rng_index:ArrayBuffer[Int]=ArrayBuffer();
    if(batch_num_per==1.0){
      for(i <- 0 to (batch_num-1)) rng_index += i//抽样样本的角标 
    }else{
      for(i <- 0 to (batch_num-1)) rng_index += math.round((rng_epooch.nextDouble()*(inputs.length-1)).toFloat)//抽样样本的角标        
    }
    //正式训练一批次的样本
    for(i <- rng_index) {
      //根据一个样本完成一次训练,迭代增量取 1/batch_num
      pre_train_da(inputs(i), lr,corruption_level,batch_num)
    }
    //完成一批次训练后计算本批次的平均交叉嫡cross_entropy_result/batch_num (cross_entropy_result内部是累加的)
    println("cross_entropy="+cross_entropy_result/batch_num) 
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
  //不训练,直接经过编码和解码过程
  def reconstruct(x: Array[Array[Array[Double]]], z: Array[Array[Array[Double]]]) {
    val y: Array[Array[Array[Double]]] = Array.ofDim[Double](n_kernel,s0,s1)
    get_hidden_values(x, y)
    get_reconstructed_input(y,z)
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
                    var rng: Random=null) {
  
  if(rng == null) rng = new Random(1234) 

  
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
                /* vision 1
                //相当于MATLAB中的convn(next_layer.d_v,rot180(next_layer.W),'full'),即完成了2维的卷积运算
                d_v(c)(tmp1+s)(tmp2+t) += next_layer.d_v(k)(tmp1)(tmp2) * next_layer.W(k)(c)(s)(t)
                //由于是max 所以没有 *dactivation_fun(input)
                */
                
                /* vision 2*/
                //相当于MATLAB中的convn(next_layer.d_v,rot180(next_layer.W),'full'),即完成了2维的卷积运算
                //其中 d_v(c)(tmp1+s)(tmp2+t) += next_layer.d_v(k)(tmp1)(tmp2) * next_layer.W(k)(c)(s)(t) 实现了convn(next_layer.d_v,next_layer.W,'full')
                //把next_layer.W(k)(c)(s)(t)换为next_layer.W(k)(c)((next_layer.kernel_size._1-1)-s)((next_layer.kernel_size._2-1)-t) 实现了rot180
                //d_v(c)(tmp1+s)(tmp2+t) += next_layer.d_v(k)(tmp1)(tmp2) * next_layer.W(k)(c)(s)(t) 
                d_v(c)(tmp1+s)(tmp2+t) += next_layer.d_v(k)(tmp1)(tmp2) * next_layer.W(k)(c)((next_layer.kernel_size._1-1)-s)((next_layer.kernel_size._2-1)-t)
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
          var rng: Random=null,
          activation:String="ReLU") {
  
  //用于初始化conv成参数
  //参考lisa lab和yusugomori
  val f_in_tmp:Int  = n_channel_in * kernel_size_in._1 * kernel_size_in._2
  val f_out_tmp:Int = (n_kernel_in * kernel_size_in._1 * kernel_size_in._2)/(pool_size_in._1*pool_size_in._2)
  //val init_a_tmp:Double=math.sqrt(6.0/(f_in_tmp + f_out_tmp)) 
  val init_a_tmp:Double=1/ math.pow(f_out_tmp,0.25)  //cnn simple
  val ConvLayer_obj:ConvLayer= new ConvLayer(input_size_in=input_size_in,
                                             n_kernel_in=n_kernel_in,
                                             kernel_size_in=kernel_size_in,
                                             _W=_W,
                                             _b=_b,
                                             init_a=init_a_tmp,
                                             n_channel_in=n_channel_in,
                                             rng=rng,
                                             activation=activation)
  val Max_PoolLayer_obj:Max_PoolLayer=new Max_PoolLayer(input_size=(ConvLayer_obj.s0,ConvLayer_obj.s1),
                                                        pre_conv_layer_n_kernel_in=ConvLayer_obj.n_kernel,
                                                        pool_size_in=pool_size_in,
                                                        rng=rng)
  
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
    /* vision 1
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
    )*/
    
    /* vision 2 */
    var init_w_2:Array[Array[Array[Array[Double]]]]=Array(
        Array(
            Array(
                Array(0.4,0.2), 
                Array(0.2,0.1)
            )            
        ),
        Array(
            Array(
                Array(0.2,0.1), 
                Array(0.1,-0.3)
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
d_v_i=conv(nextlayer.d_v,rot180(nextlayer.w),'full')
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
0.0	0.0	0.0	-104.49666567769285	
0.0	-16.570779564884706	0.0	0.0	
-19.85805972596852	0.0	0.0	0.0	
0.0	0.0	-99.48737429897888	0.0	
W_add_0_0:
-1153.8546296767536	-913.4417504092287	-673.0288711417038	-432.6159918741788	
-1153.8546296767536	-913.4417504092287	-673.0288711417038	-432.6159918741788	
-1153.8546296767536	-913.4417504092287	-673.0288711417038	-432.6159918741788	
-1153.8546296767536	-913.4417504092287	-673.0288711417038	-432.6159918741788	
W_add_0_0=rot180(convn(x,rot180(d_v)),'valid'))
     * */    
                                                             
    //pre_train与训练
    //使用mnist数据集进行训练
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
    val n_epochs: Int = 200
    val convpool_obj: ConvPoolLayer = new ConvPoolLayer(input_size_in=(height,width),
                                                 n_kernel_in=50,
                                                 kernel_size_in=(5,5),
                                                 pool_size_in=(2,2),
                                                 _W=null,
                                                 _b=null,
                                                 n_channel_in=1,
                                                 rng=null,
                                                 activation="ReLU")

    // train
    var epoch: Int = 0
    //批量训练 training_epochs次
    for(epoch <- 0 until n_epochs) {
      println("第"+epoch+"次迭代:")
      convpool_obj.ConvLayer_obj.pre_train_da_batch(train_X, learning_rate, 0.1, 0.01)
      if(epoch==(n_epochs-1) || epoch==math.round(n_epochs.toDouble/4.0) || epoch==math.round(2.0*n_epochs.toDouble/4.0) || epoch==math.round(3.0*n_epochs.toDouble/4.0)){
        //记录权值w
        convpool_obj.ConvLayer_obj.save_w("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//Conv_mnist_module"+epoch+".txt")
      }
    } 
                                                             
  }   
  
}