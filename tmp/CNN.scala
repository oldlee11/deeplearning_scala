package dp_process

import scala.util.Random
import scala.math

import java.io.File; 
import java.io._
import scala.io.Source

import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

/*
 * input_size           处理图片的长和高
 * n_channel            图片组成的个数,一般一个图片是以RGB形式给出的,所以一般=3
 * output_size          分类个数
 * n_kernel_Array       每一个 ConvPoolLayer层内卷积层的n_kernel
 * kernel_size_Array    每一个 ConvPoolLayer层内卷积层的kernel_size
 * pool_size_Array      每一个 ConvPoolLayer层内池化层的pool_size
 * activation           每一个 ConvPoolLayer层内卷积层的非线性函数
 * */
class CNN(input_size:(Int,Int),
          output_size:Int,
          n_kernel_Array:Array[Int],
          kernel_size_Array:Array[(Int,Int)],
          pool_size_Array:Array[(Int,Int)],
          n_hidden:Int,
          n_channel:Int=3,
          _rng: Random=null,
          activation:String="ReLU",
          activation_mlp:String="tanh") {
  
  var rng:Random=if(_rng == null) new Random(1234) else _rng
    
  //ConvPoolLayer层个数
  val n_ConvPoolLayer=n_kernel_Array.length
  val ConvPoolLayer_layers:Array[ConvPoolLayer]= new Array[ConvPoolLayer](n_ConvPoolLayer)
  //建立ConvPoolLayer_layers
  for(i <-0 until n_ConvPoolLayer){
    if(i==0){  
      ConvPoolLayer_layers(i)=new ConvPoolLayer(input_size_in=input_size,
                                                n_kernel_in=n_kernel_Array(i),
                                                kernel_size_in=kernel_size_Array(i),
                                                pool_size_in=pool_size_Array(i),
                                                n_channel_in=n_channel,
                                                _rng=rng,
                                                activation=activation)  
    }else{
      ConvPoolLayer_layers(i)=new ConvPoolLayer(input_size_in=(ConvPoolLayer_layers(i-1).Max_PoolLayer_obj.s0,ConvPoolLayer_layers(i-1).Max_PoolLayer_obj.s1),
                                                n_kernel_in=n_kernel_Array(i),
                                                kernel_size_in=kernel_size_Array(i),
                                                pool_size_in=pool_size_Array(i),
                                                n_channel_in=ConvPoolLayer_layers(i-1).Max_PoolLayer_obj.pre_conv_layer_n_kernel,
                                                _rng=rng,
                                                activation=activation)        
    }
  }
  
  //建立输出层  一成hidden+一层logistic
  val mlp_layer = new Dropout(n_in=ConvPoolLayer_layers(n_ConvPoolLayer-1).Max_PoolLayer_obj.pre_conv_layer_n_kernel*ConvPoolLayer_layers(n_ConvPoolLayer-1).Max_PoolLayer_obj.s0*ConvPoolLayer_layers(n_ConvPoolLayer-1).Max_PoolLayer_obj.s1, 
                              hidden_layer_sizes=Array(n_hidden),
                              n_out=output_size,
                              activation=activation_mlp)
  
  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)
    x:一个样本的x数据,据取值为[0,1]之间  样本是一个3维数据形式=n_channel*input_size(0)*input_size(1)
    y:样本的标准输出
    lr:参数迭代的学习率
    batch_num 是一次批量迭代使用的样本数,由于每一次训练时,样本仅输入1个,参数(W,vbias,hbias)迭代计算时：W(t+1)=W(t)+lr*(局部误差/batch_num)
                                                 即每一次迭代仅仅更新1/batch_num倍,如果batch_num=训练集样本个数,则当所有训练集都迭代一遍后则 为 W(t+1)=W(t)+lr*(局部误差)
                                                如果batch_num=1则表示每个样本迭代一次 W(t+1)=W(t)+lr*(局部误差) 即不是批量迭代而是随机迭代了  
  */
  def train(x: Array[Array[Array[Double]]],y:Array[Int], lr: Double,batch_num:Int,dropout:Boolean=true,alpha:Double=0.0,debug:Boolean=false) { 
    /*
     * step1 forward ConvPoolLayer_layers
     */   
    var layer_inputs:ArrayBuffer[Array[Array[Array[Double]]]]=ArrayBuffer()
    var layer_input:Array[Array[Array[Double]]]=Array()    
    for(i <-0 until n_ConvPoolLayer){
      if(i==0) { 
        layer_input=x       
      }
      layer_inputs +=layer_input
      ConvPoolLayer_layers(i).cnn_forward(layer_input)
      layer_input=ConvPoolLayer_layers(i).Max_PoolLayer_obj.pooled_input
    }
    
    /*
     * step2 forward & backward log_layer
    */
    val layer_input_copy_one:Array[Double]=flatten(layer_input)//打平
    mlp_layer.train(x=layer_input_copy_one,
                    y=y,
                    lr=lr,
                    alpha=alpha, 
                    batch_num=batch_num,
                    dropout=true, 
                    p_dropout=0.2)
    if(debug){
      def see_w_max_min(w_in:Array[Array[Array[Array[Double]]]],max_min:String):Double={
        var result:Double=0.0;
        for(i<-0 until w_in.length){
          for(j<-0 until w_in(0).length){
            for(k<-0 until w_in(0)(0).length){
              for(m<-0 until w_in(0)(0)(0).length){
                if(max_min=="max")
                  result=math.max(w_in(i)(j)(k)(m),result)
                else if(max_min=="min")
                  result=math.min(w_in(i)(j)(k)(m),result)
                else if(max_min=="sum")
                  result +=w_in(i)(j)(k)(m)
              }
            }
          }
        }
        result
      }
      def see_d_v_max_min(w_in:Array[Array[Array[Double]]],max_min:String):Double={
        var result:Double=0.0;
        for(i<-0 until w_in.length){
          for(j<-0 until w_in(0).length){
            for(k<-0 until w_in(0)(0).length){
                if(max_min=="max")
                  result=math.max(w_in(i)(j)(k),result)
                else if(max_min=="min")
                  result=math.min(w_in(i)(j)(k),result)
                else if(max_min=="sum")
                  result +=w_in(i)(j)(k)
            }
          }
        }
        result
      }      
      print("max flatten out are:"+layer_input_copy_one.reduce(math.max(_,_))+"\n")
      print("min flatten out are:"+layer_input_copy_one.reduce(math.min(_,_))+"\n") 
      print("avg flatten out are:"+layer_input_copy_one.reduce(_ + _)/layer_input_copy_one.length+"\n") 
      for(i<-0 until n_kernel_Array.length){
        print("max convpoollayer_"+i+" w are:"+see_w_max_min(ConvPoolLayer_layers(i).ConvLayer_obj.W,"max")+"\n")
        print("min convpoollayer_"+i+" w are:"+see_w_max_min(ConvPoolLayer_layers(i).ConvLayer_obj.W,"min")+"\n")   
        //print("sum convpoollayer_"+i+" w are:"+see_w_max_min(ConvPoolLayer_layers(i).ConvLayer_obj.W,"sum")+"\n")    
        print("sum_per_out convpoollayer_"+i+" w are:"+see_w_max_min(ConvPoolLayer_layers(i).ConvLayer_obj.W,"sum")/ConvPoolLayer_layers(i).f_out_tmp+"\n")  
        print("max convpoollayer_"+i+" d_v are:"+see_d_v_max_min(ConvPoolLayer_layers(i).ConvLayer_obj.d_v,"max")+"\n")
        print("min convpoollayer_"+i+" d_v are:"+see_d_v_max_min(ConvPoolLayer_layers(i).ConvLayer_obj.d_v,"min")+"\n")  
      }
      print("max hidden_layer w:"+mlp_layer.hidden_layers(0).W.reduce(_++_).reduce(math.max(_,_))+"\n");//debug
      print("min hidden_layer w:"+mlp_layer.hidden_layers(0).W.reduce(_++_).reduce(math.min(_,_))+"\n");//debug
      //print("sum hidden_layer w:"+mlp_layer.hidden_layers(0).W.reduce(_++_).reduce(_+_)+"\n");//debug
      print("sum_per_out hidden_layer w:"+mlp_layer.hidden_layers(0).W.reduce(_++_).reduce(_+_)/mlp_layer.hidden_layers(0).n_out+"\n");//debug
      print("max hidden_layer d_v:"+mlp_layer.hidden_layers(0).d_v.reduce(math.max(_,_))+"\n");//debug
      print("min hidden_layer d_v:"+mlp_layer.hidden_layers(0).d_v.reduce(math.min(_,_))+"\n");//debug
      print("max log_layer w:"+mlp_layer.log_layer.W.reduce(_++_).reduce(math.max(_,_))+"\n");//debug
      print("min log_layer w:"+mlp_layer.log_layer.W.reduce(_++_).reduce(math.min(_,_))+"\n");//debug
      //print("sum log_layer w:"+mlp_layer.log_layer.W.reduce(_++_).reduce(_+_)+"\n");//debug   
      print("sum_per_out log_layer w:"+mlp_layer.log_layer.W.reduce(_++_).reduce(_+_)/mlp_layer.log_layer.n_out+"\n");//debug     
      print("max out:"+mlp_layer.predict(layer_input_copy_one).reduce(math.max(_,_))+"\n");//debug
      print("min out:"+mlp_layer.predict(layer_input_copy_one).reduce(math.min(_,_))+"\n");//debug
      print("max hidden_layer d_v:"+mlp_layer.log_layer.d_y.reduce(math.max(_,_))+"\n");//debug
      print("min hidden_layer d_v:"+mlp_layer.log_layer.d_y.reduce(math.min(_,_))+"\n");//debug      
    }    
    
    /*
     * step3 backward ConvPoolLayer_layers
     */  
    for(i <- (0 until n_ConvPoolLayer).reverse){
      if (i == n_ConvPoolLayer-1){
        //下一层ConvPoolLayer_layers(i+1)是Hidden
        ConvPoolLayer_layers(i).cnn_backward_2(next_layer=mlp_layer,x=layer_inputs(i),batch_num=batch_num,lr=lr,alpha=alpha)
      } else{
        //下一层ConvPoolLayer_layers(i+1)是ConvPoolLayer_layers
        ConvPoolLayer_layers(i).cnn_backward_1(next_layer=ConvPoolLayer_layers(i+1),x=layer_inputs(i),batch_num=batch_num,lr=lr,alpha=alpha)
      }  
    }
  }
  
  /*训练,使用训练集做一个批次的BP训练
    lr:参数迭代的学习率
    inputs_x 训练集输入x  是4维数组=样本个数*n_channel*input_size(0)*input_size(1)
    inputs_y 训练集输出y
   * */  
  def train_batch(inputs_x: Array[Array[Array[Array[Double]]]], inputs_y: Array[Array[Int]],lr: Double,batch_num_per:Double=1.0,alpha:Double=0.0,save_module_path:String="",debug:Boolean=false)={
    //抽取样本个数
    val batch_num:Int=if(batch_num_per==1.0){
      inputs_x.length
    }else{
      math.round((inputs_x.length*batch_num_per).toFloat)//每次批量训练样本数
    }
    mlp_layer.log_layer.cross_entropy_result=0.0//每次批量开始时,把上次的交叉信息嫡清零  
    
    //完成一次批量训练
    val rng_epooch:Random=new Random()//每次生成一个种子
    val rng_index:ArrayBuffer[Int]=ArrayBuffer();
    if(batch_num_per==1.0){
      for(i <- 0 to (batch_num-1)) rng_index += i//抽样样本的角标 
    }else{
      for(i <- 0 to (batch_num-1)) rng_index += math.round((rng_epooch.nextDouble()*(inputs_x.length-1)).toFloat)//抽样样本的角标        
    }
    //正式训练一批次的样本
    for(i <- rng_index) {
      //根据一个样本完成一次训练,迭代增量取 1/batch_num
      if(debug){
        print("\n")//debug
      }
      train(x=inputs_x(i),y=inputs_y(i), lr=lr,batch_num=batch_num,debug=debug,alpha=alpha)
    }
    //完成一批次训练后计算本批次的平均交叉嫡cross_entropy_result/batch_num (cross_entropy_result内部是累加的)
    println("cross_entropy="+mlp_layer.log_layer.cross_entropy_result/batch_num)    
    
    /*//保存第i层da的参数
    if(! (save_module_path=="")){
      ConvPoolLayer_layers(i).save_w(save_module_path+"cnn_module_ConvPoolLayer"+i+".txt") //debug 
    } */   
  }
  
  //预测一个样本
  //注意 没有返回值  ,直接改变y
  def predict(x: Array[Array[Array[Double]]]):Array[Double]={
    var layer_input:Array[Array[Array[Double]]] = x
    for(i <- 0 until n_ConvPoolLayer){
      layer_input=ConvPoolLayer_layers(i).output(layer_input)
      //ConvPoolLayer_layers(i).cnn_forward(layer_input)
      //layer_input=ConvPoolLayer_layers(i).Max_PoolLayer_obj.pooled_input
    }
    
    val layer_input_copy_one=flatten(layer_input)//打平  
    mlp_layer.predict(layer_input_copy_one)
  }  
  
  
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
}


object CNN {
  
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
    val classifier = new  CNN(input_size=(9,9),output_size=n_out,n_kernel_Array=Array(15,20),kernel_size_Array=Array((2,2),(3,3)),pool_size_Array=Array((2,2),(2,2)),n_channel=2,n_hidden=20,_rng=null,activation="ReLU",activation_mlp="tanh")//n_epochs=250 alpha=0.0 learning_rate *=0.99  lr=0.1
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
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate, batch_num_per=1.0, alpha=0.0,save_module_path="",debug=false)
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
0.95984 0.02178 0.01838 
0.01030 0.00744 0.98226 
0.09209 0.85717 0.05074 
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
    val n_epochs: Int = 300
    
    //单层
    //val classifier = new  CNN(input_size=(height,width),output_size=10,n_kernel_Array=Array(20),kernel_size_Array=Array((9,9)),pool_size_Array=Array((2,2)),n_channel=1,n_hidden=84,rng=null,activation="ReLU",activation_mlp="tanh")//lr=0.1 alpha=0.0 learning_rate不变     迭代次数200+ 
                                                                                                                                                                                                                                     //hidden val a: Double = 4 * math.sqrt(6.0/(n_in + n_out))
                                                                                                                                                                                                                                     //cnn    val init_a_tmp:Double=1/ math.pow(f_out_tmp,0.25)
                                                                                                                                                                                                                                     //正确率=87.8%
    //lenet5
    val classifier = new  CNN(input_size=(height,width),output_size=10,n_kernel_Array=Array(6,16,120),kernel_size_Array=Array((5,5),(5,5),(4,4)),pool_size_Array=Array((2,2),(2,2),(1,1)),n_channel=1,n_hidden=84,_rng=null,activation="ReLU",activation_mlp="tanh")//lr=0.1 alpha=0.0 learning_rate*=0.99     迭代次数200+ 
                                                                                                                                                                                                                                                                   //hidden val a: Double = 4 * math.sqrt(6.0/(n_in + n_out))
                                                                                                                                                                                                                                                                   //cnn    val init_a_tmp:Double=math.sqrt(6.0/(f_in_tmp + f_out_tmp)) 或者=1/ math.pow(f_out_tmp,0.25)
                                                                                                                                                                                                                                                                   //正确率=89。3%
    
    // train
    var epoch: Int = 0
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate, batch_num_per=0.01,alpha=0.0, save_module_path="",debug=false)
      learning_rate *=0.99
    } 
    
    /*
     * test
     * */
    val filePath_test:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt"  
    val test_X:Array[Array[Array[Array[Double]]]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>{val tmp:Array[Array[Double]]=Array.ofDim[Double](height,width);for(i <- 0 until height){for(j <-0 until width){tmp(i)(j)=x._2(i*width+j)}};Array(tmp)})
    val test_N: Int = 1000//test_X.length
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
    //test_CNN_simple()//ok
    train_test_mnist()//--没成功
  }   
}  