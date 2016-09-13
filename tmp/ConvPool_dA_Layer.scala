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
class Conv_dA_Layer(input_size_in:(Int,Int),
                    n_kernel_in:Int,
                    kernel_size_in:(Int,Int),
                    init_a:Double=0.0,
                    _W:Array[Array[Array[Array[Double]]]]=null,
                    _b:Array[Double]=null,
                    n_channel_in:Int=3,
                    _rng: Random=null,
                    activation:String="ReLU") extends ConvLayer(input_size_in,
                    n_kernel_in,
                    kernel_size_in,
                    init_a,
                    _W,
                    _b,
                    n_channel_in,
                    _rng,
                    activation){

  /*  之前的代码一点不变
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
                /* 参考 convolve_forward  
                 * 相当于MATLAB中的convn(X,rot180(W),'viald')*/                
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
    //遍历y
    for(k <- 0 until n_kernel){
      for(i <- 0 until s0){ 
        for(j <- 0 until s1){
          //遍历k内核
          for(c <- 0 until n_channel){
            for(s <- 0 until kernel_size._1){
              for(t <- 0 until kernel_size._2){
                /* 参考 maxpoollayer_backward_1  
                 * 相当于MATLAB中的convn(y,W,'full')*/
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
      /* 计算 L_vbias 并对vbias迭代 */
      val L_vbias: Array[Array[Array[Double]]] = Array.ofDim[Double](x.length,x(0).length,x(0)(0).length)//x和z的误差
      for(i<- 0 until x.length){
        for(j<- 0 until x(0).length){
          for(k<- 0 until x(0)(0).length){
            L_vbias(i)(j)(k)=x(i)(j)(k)-z(i)(j)(k)
            vbias(i) += lr * L_vbias(i)(j)(k)/batch_num
          }
        }
      }
      /* 计算L_hbias 并对b迭代 */
      val L_hbias:Array[Array[Array[Double]]] = Array.ofDim[Double](y.length,y(0).length,y(0)(0).length)
      for(k <- 0 until n_kernel){
        for(i <- 0 until s0){ 
          for(j <- 0 until s1){
            for(c <- 0 until n_channel){
              for(s <- 0 until kernel_size._1){
                for(t <- 0 until kernel_size._2){
                  L_hbias(k)(i)(j) += W(k)(c)(s)(t) * L_vbias(c)(i+s)(j+t)//L_vbias向前传播
                }
              }
            }
            L_hbias(k)(i)(j) =L_hbias(k)(i)(j) * utils.dsigmoid(y(k)(i)(j))
            b(k) += lr * L_hbias(k)(i)(j) / batch_num
          }
        }
      }      
      for (c<- 0 until n_channel){
        for(i<- 0 until s0){
          for(j<- 0 until s1){
            for(k <- 0 until n_kernel){
              for(s <- 0 until kernel_size._1){
                for(t <- 0 until kernel_size._2){
                  /*参考convolve_backward
                   * convn(tilde_x,rot180(L_hbias),'valid')  +  convn(L_vbias,rot180(y),'valid')*/
                  W(k)(c)(s)(t) +=lr *( tilde_x(c)(i+s)(j+t)*L_hbias(k)(i)(j)+L_vbias(c)(i+s)(j+t)*y(k)(i)(j) )/batch_num
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

  //不训练,直接经过编码和解码过程
  def reconstruct(x: Array[Array[Array[Double]]], z: Array[Array[Array[Double]]]) {
    val y: Array[Array[Array[Double]]] = Array.ofDim[Double](n_kernel,s0,s1)
    get_hidden_values(x, y)
    get_reconstructed_input(y,z)
  }
}  

/*
 * ConvPoolLayer=Conv_dA_Layer=>Max_PoolLayer
 * input_size_in:conv卷积层输入
 * n_kernel_in:conv卷积层的卷积核个数
 * kernel_size_in:conv卷积层的卷积核长和高
 * pool_size_in:maxpool池化层的核长和高
 * init_a_in 用于conv卷积层的初始化
 * _W 初始化conv卷积层的w系数
 * _b 初始化conv卷积层的b系数
 * activation  conv卷积层的非线性处理函数
 * */
class ConvPool_dA_Layer(input_size_in:(Int,Int),
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
  
  val ConvLayer_obj:Conv_dA_Layer= new Conv_dA_Layer(input_size_in=input_size_in,
                                             n_kernel_in=n_kernel_in,
                                             kernel_size_in=kernel_size_in,
                                             _W=_W,
                                             _b=_b,
                                             init_a=this.init_a_tmp,
                                             n_channel_in=n_channel_in,
                                             _rng=this.rng,
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

object ConvPool_dA_Layer{
  def main(args: Array[String]) {                                              
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
    val convpool_obj:ConvPool_dA_Layer = new ConvPool_dA_Layer(input_size_in=(height,width),
                                                 n_kernel_in=50,
                                                 kernel_size_in=(5,5),
                                                 pool_size_in=(2,2),
                                                 _W=null,
                                                 _b=null,
                                                 n_channel_in=1,
                                                 _rng=null,
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

