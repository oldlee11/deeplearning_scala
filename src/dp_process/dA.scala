package dp_process

import scala.util.Random


/*
 * 输入时[0-1]之间的连续变量,并进行编解码自学习(非监督学习)
 * */
import scala.math
import scala.math.log

import java.io.File; 
import java.io._
import scala.io.Source


import scala.collection.mutable.ArrayBuffer    //用于建立可变的array


/*
 * n_visible:可视层的单元个数
 * n_hidden:隐藏层的单元个数
 * _W 可视层和隐层层之间的权重系数 默认为null空
 * _hbias:隐层层的偏置,默认为null空
 * _vbias:可视层的偏置,默认为null空
 * rng:随机种子,默认为null空
 * */
class dA(val n_visible: Int, val n_hidden: Int,
         _W: Array[Array[Double]]=null, _hbias: Array[Double]=null, _vbias: Array[Double]=null,
         var rng: Random=null) {
  
  var W: Array[Array[Double]] = Array.ofDim[Double](n_hidden, n_visible)//隐藏层和可视层间的权重系数 w权重是 n_hidden行  x n_visible列的 Double 型2维矩阵
  var hbias: Array[Double] = new Array[Double](n_hidden)//每个隐藏层单元的偏置 hbias是 n_hidden长度的Double 型1维向量
  var vbias: Array[Double] = new Array[Double](n_visible)//每个可视层单元的偏置 n_visible是 n_hidden长度的Double 型1维向量
  var cross_entropy_result:Double=0.0

  if(rng == null) rng = new Random(1234)//如果没有输入随机种子,则设置为Random(1234)

  init_w_module();
  if(_W == null) {
  } else {
    //如果_W不为空则使用_W初始化W
    W = _W
  }

  if(_hbias == null) {
  } else {
    //如果_hbias不为空则使用_hbias初始化hbias
    hbias = _hbias
  }

  if(_vbias == null) {
  } else {
    //如果_vbias不为空则使用_vbias初始化vbias
    vbias = _vbias
  }

  //定义uniform均值分布的随机数产生函数
  //每次仅产生一个随机数
  def uniform(min: Double, max: Double): Double = rng.nextDouble() * (max - min) + min
  
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

  //非线性映射函数=1/(1+E(-x次幂))
  def sigmoid(x: Double): Double = 1.0 / (1.0 + math.pow(math.E, -x))

  //把一维向量x对根据二项分布p 加入噪音,最后生成tilde_x
  //如果原始数据=0,则依旧=0
  //如果原始数据不等于0,则=binomial(1, p)   取值为0或1的值,随机生成1的概率=p
  def get_corrupted_input(x: Array[Double], tilde_x: Array[Double], p: Double) {
    var i: Int = 0;
    for(i <- 0 until n_visible) {
      if(x(i) == 0) {
        tilde_x(i) = 0.0;
      } else {
        tilde_x(i) = binomial(1, p)*x(i)//有p的概率 tilde_x(i) =x(i)*1=x(i) ,有1-p的概率  tilde_x(i) =x(i)*0=0
      }
    }
  }

  // Encode 编码过程：v-->h
  // 把可视层(v)内的单元的值 经过v->h连接的系数W加权求和并加上隐藏层单元的偏置,最后经过sigmoid映射为[0,1]数值=隐藏层(h)内每个单元的数值=y
  def get_hidden_values(x: Array[Double], y: Array[Double]) {
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_hidden) {
      y(i) = 0
      for(j <- 0 until n_visible) {
        y(i) += W(i)(j) * x(j)
      }
      y(i) += hbias(i)
      y(i) = sigmoid(y(i))
    }
  }

  // Decode  解码过程由:h-->v
  // 把隐藏层(h)内的单元的值 经过h->v连接的系数W加权求和并加上可视层单元的偏置,最后经过sigmoid映射为[0,1]数值=可视层(v)内每个单元的数值=z
  def get_reconstructed_input(y: Array[Double], z: Array[Double]) {
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_visible) {
      z(i) = 0
      for(j <- 0 until n_hidden) {
        z(i) += W(j)(i) * y(j)
      }
      z(i) += vbias(i)
      z(i) = sigmoid(z(i))
    }
  }

  
  def train_batch(inputs: Array[Array[Double]], lr: Double, corruption_level: Double,batch_num_per:Double):Unit={
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
      train(inputs(i), lr,corruption_level,batch_num)
    }
    //完成一批次训练后计算本批次的平均交叉嫡cross_entropy_result/batch_num (cross_entropy_result内部是累加的)
    println("cross_entropy="+cross_entropy_result/batch_num)         
  }
  
  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)
    x:一个样本的x数据,据取值为[0,1]之间
    lr:参数迭代的学习率
    corruption_level:加噪音的比例  =[0,1],值越高,加入噪音个数约多(原始数据输入由x->0)
    batch_num 是一次批量迭代使用的样本数,由于每一次训练时,样本仅输入1个,参数(W,vbias,hbias)迭代计算时：W(t+1)=W(t)+lr*(局部误差/batch_num)
                                                 即每一次迭代仅仅更新1/batch_num倍,如果batch_num=训练集样本个数,则当所有训练集都迭代一遍后则 为 W(t+1)=W(t)+lr*(局部误差)
                                                如果batch_num=1则表示每个样本迭代一次 W(t+1)=W(t)+lr*(局部误差) 即不是批量迭代而是随机迭代了  
  */
  def train(x: Array[Double], lr: Double, corruption_level: Double,batch_num:Int) {
    var i: Int = 0
    var j: Int = 0

    val tilde_x: Array[Double] = new Array[Double](n_visible)
    val y: Array[Double] = new Array[Double](n_hidden)
    val z: Array[Double] = new Array[Double](n_visible)

    val L_vbias: Array[Double] = new Array[Double](n_visible)
    val L_hbias: Array[Double] = new Array[Double](n_hidden)

    val p: Double = 1 - corruption_level

    //  x--->加噪音--->tilde_x
    get_corrupted_input(x, tilde_x, p)
    // tilde_x----->编码---->y---->解码---->z
    get_hidden_values(tilde_x, y)//完成一次编码,输出=y
    get_reconstructed_input(y, z)//完成一次解码,输出=z
    this.cross_entropy_result=this.cross_entropy_result + cross_entropy(x,z)//衡量输入和经过编解码后的输出数据之间的相似度(使用KL散度,即相对信息嫡)

    // vbias迭代
    for(i <- 0 until n_visible) {
      L_vbias(i) = x(i) - z(i)
      vbias(i) += lr * L_vbias(i) / batch_num
    }

    // hbias迭代
    for(i <- 0 until n_hidden) {
      L_hbias(i) = 0
      for(j <- 0 until n_visible) {
        L_hbias(i) += W(i)(j) * L_vbias(j)
      }
      L_hbias(i) *= y(i) * (1 - y(i))//得到了解码层反向输出的误差L_h1,  y * (1 - y) 是simgoid的导数
      hbias(i) += lr * L_hbias(i) / batch_num
    }

    // W迭代
    for(i <- 0 until n_hidden) {
      for(j <- 0 until n_visible) {
        W(i)(j) += lr * (L_hbias(i) * tilde_x(j) + L_vbias(j) * y(i)) / batch_num
      }
    }
  }

  //不训练,直接经过编码和解码过程
  def reconstruct(x: Array[Double], z: Array[Double]) {
    val y: Array[Double] = new Array[Double](n_hidden)
    get_hidden_values(x, y)
    get_reconstructed_input(y,z)
  }

  //使用交叉信息嫡cross-entropy衡量样本输入和经过编解码后输出的相近程度
  //值在>=0 当为0时 表示距离接近
  //每次批量迭代后该值越来越小
  def cross_entropy(x: Array[Double], z: Array[Double]):Double={
   -1.0* (0 until x.length).map(i=>x(i)*log(z(i))+(1-x(i))*log(1-z(i))).reduce(_+_)
  }
  
  //记录系数
  def save_w(file_module_in:String):Unit={
   val writer_module = new PrintWriter(new File(file_module_in)) 
   //write w
   writer_module.write(W.map(w_i=>w_i.mkString(sep=",")).mkString(sep="=sepi="))
   writer_module.write("\n")
   //write vbias
   writer_module.write(vbias.mkString(sep=","))
   writer_module.write("\n")
   //write hbias
   writer_module.write(hbias.mkString(sep=","))
   writer_module.close()
  }
  
  //读取 save_w输出的模型txt
  //并修改dA的W    vbias    hbias
  def read_w_module(file_module_in:String):Unit={
    var result:ArrayBuffer[String]=ArrayBuffer();
    Source.fromFile(file_module_in).getLines().foreach(line=>result+=line) 
    val result_w:Array[Array[Double]]=result(0).split("=sepi=").map(x=>x.split(",").map(y=>y.toDouble))
    for(i <- 0 until n_hidden)
      for(j <- 0 until n_visible)
        W(i)(j) = result_w(i)(j)
    vbias=result(1).split(",").map(x=>x.toDouble)
    hbias=result(2).split(",").map(x=>x.toDouble)
  }
  
  //初始化清空系数w
  def init_w_module():Unit={
    var i: Int = 0
    var j: Int = 0
    //val a: Double = 1 / n_visible                           //方案1 见yusugomori
    val a: Double = 4 * math.sqrt(6.0/(n_hidden + n_visible)) //方案2 见lisa DeepLearningTutorials-master
    for(i <- 0 until n_hidden)
      for(j <- 0 until n_visible)
        W(i)(j) = uniform(-a, a)
    //则初始化hbias=0
    i= 0
    for(i <- 0 until n_hidden) hbias(i) = 0
    //初始化vbias=0
    i= 0
    for(i <- 0 until n_visible) vbias(i) = 0     
  }
  
  //把系数转化为图片
  def see_train_w_module(out_path:String,width:Int,height:Int):Unit={
    //权值的最大值
    var max_value:Double=W.map(x=>x.reduce((x,y)=>if(x>=y) x else y)).reduce((x,y)=>if(x>=y) x else y)
    for(i:Int <- 0 until W.length){
      val tmp:Array[Int]=W(i).map(y=>if(y<0.0) 0 else (255.0*y/max_value).toInt)
      dp_utils.gen_pict.gen_pict_use_one_fun(tmp,width,height,out_path+"/"+i+".jpg")
    }
  }
  
}

object dA {
  //简单测试
  def test_dA_simple() {
    val rng: Random = new Random(123)
    var learning_rate: Double = 0.1
    val corruption_level: Double = 0.3
    val training_epochs: Int = 500
    val train_N: Int = 10//10个样本
    val test_N: Int = 2//2个样本
    val n_visible: Int = 20//可视层20个单元=每个样本中有20个变量
    val n_hidden: Int = 5//隐藏层5个单元=提取5个特征
    val train_X: Array[Array[Double]] = Array(
      Array(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      Array(1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      Array(1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      Array(1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      Array(0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
      Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1),
      Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1),
      Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1),
      Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0)
    )
    val da: dA = new dA(n_visible, n_hidden, rng=rng)
    var i: Int = 0
    var j: Int = 0
    // train
    var epoch: Int = 0
    //批量训练 training_epochs次
    for(epoch <- 0 until training_epochs) {
      //清空 重新计算 本轮批量迭代的cross_entropy
      da.cross_entropy_result=0.0
      println("进行第"+epoch+"次批量训练")      
      //完成一次批量训练
      /*for(i <- 0 until train_N) {
        //根据一个样本完成一次训练,迭代增量取 1/N    
        da.train(train_X(i), learning_rate, corruption_level,train_N)
      }*/
      da.train_batch(train_X,learning_rate, corruption_level,1.0)
      println("cross_entropy="+da.cross_entropy_result/train_N)
    }
    // test data
    val test_X: Array[Array[Double]] = Array(
      Array(1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0)
    )
    val reconstructed_X: Array[Array[Double]] = Array.ofDim[Double](test_N, n_visible)
    for(i <- 0 until test_N) {
      da.reconstruct(test_X(i), reconstructed_X(i))
      println("样本"+i+":\n")
      println("    输入数据=")
      for(j <- 0 until n_visible) {
        printf("%.5f ", test_X(i)(j))
      }
      println()
      println("    预测数据=")
      for(j <- 0 until n_visible) {
        printf("%.5f ", reconstructed_X(i)(j))
      }      
      println()
    /*
样本1:前10个每个值>0.5即=1,后10个每个值<0.5即=0
0.78975 0.79424 0.79279 0.98409 0.79107 0.79147 0.98422 0.98400 0.79307 0.78961 0.00963 0.01002 0.00980 0.01515 0.00969 0.00957 0.01536 0.01498 0.00963 0.00975 
样本1:前10个每个值<0.5即=0,后10个每个值>0.5即=1
0.05206 0.05318 0.05223 0.05953 0.05204 0.05199 0.05974 0.05993 0.05179 0.05202 0.78386 0.78444 0.78435 0.94192 0.78414 0.78435 0.94142 0.94169 0.78462 0.78375 
     */      
    }   
    //保存系数
    da.save_w("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//dA_simple_module.txt")
    //观看da系数的图像
    da.see_train_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//",10,2)
    //清空系数
    da.init_w_module()
    //读取系数
    da.read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//dA_simple_module.txt")
    //再次保存系数
    da.save_w("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//dA_simple_module2.txt")
  }
  
  //使用mnist数据集进行训练
  def dA_train_mnist():Unit={
    //读取训练集数据
    val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/training_data.txt"
    val train_X:Array[Array[Double]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>x._2)
    val train_N: Int = train_X.length
    
    //设置da的基础参数并建立da
    val rng: Random = new Random(123)
    var learning_rate: Double = 0.1//学习率,步长
    val corruption_level: Double = 0.1//加入10%的噪音
    val training_epochs: Int = 200//批量迭代次数=500
    val n_visible: Int = 28 * 28//mnist是28X28的像素图片,所以每个可视层的单元对应一个位置的像素值
    val n_hidden: Int = 30//隐藏层500个单元=提取500个特征    
    //由于训练集太多,这里每次批量的个数<>train_N 而是随机抽取一定比例的样本,比例=batch_num_per
    val batch_num_per:Double=0.3  //每次批量训练 提取训练样本的比例   =1.0表示不抽样   =20%表示每次抽样20%的样本做一次批量训练
    val da: dA = new dA(n_visible, n_hidden, rng=rng)
    
    // train
    var epoch: Int = 0
    //批量训练 training_epochs次
    for(epoch <- 0 until training_epochs) {
      println("第"+epoch+"次迭代:")
      da.train_batch(train_X,learning_rate, corruption_level,batch_num_per)
      if(epoch==(training_epochs-1) || epoch==math.round(training_epochs.toDouble/4.0) || epoch==math.round(2.0*training_epochs.toDouble/4.0) || epoch==math.round(3.0*training_epochs.toDouble/4.0)){
        //记录权值w
        da.save_w("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//dA_mnist_module"+epoch+".txt")
      }
    }  
    //观看da系数的图像
    da.see_train_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//",28,28)
  }
  
  //使用mnist数据集进行测试
  def dA_test_mnist():Unit={
    //载入测试集数据
    val filePath_test:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt"
    val test_X:Array[Array[Double]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>x._2)
    val test_N: Int = test_X.length
    
    //设置da的基础参数并建立da
    val n_visible: Int = 28 * 28//mnist是28X28的像素图片,所以每个可视层的单元对应一个位置的像素值
    val n_hidden: Int = 30//隐藏层500个单元=提取500个特征        
    val rng: Random = new Random(123)
    val da: dA = new dA(n_visible, n_hidden, rng=rng)
    //使用训练时生成的权值,初始化测试da的权值w
    da.read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//dA_mnist_module50.txt")
    da.see_train_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//",28,28)
    
    //test
    val writer_test_out = new PrintWriter(new File("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//test//dA_mnist_test.txt"))
    val reconstructed_X: Array[Array[Double]] = Array.ofDim[Double](test_N, n_visible)
    for(i <- 0 until test_N) {
      da.reconstruct(test_X(i), reconstructed_X(i))
      writer_test_out.write("样本"+i+":\n")
      writer_test_out.write("    输入数据=")
      for(j <- 0 until n_visible) {
        writer_test_out.write(test_X(i)(j).toString()+",")
      }
      //生成图片
      dp_utils.gen_pict.gen_pict_use_one_fun(test_X(i).map(x=>math.round(x*255.0).toInt).toArray, 28, 28,"D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/test/da_mnist_test/dA_mnist_test_"+i+".jpg")
      writer_test_out.write("\n")
      writer_test_out.write("    预测数据=")      
      for(j <- 0 until n_visible) {
        writer_test_out.write(reconstructed_X(i)(j).toString()+",")
      }
      //生成图片
      dp_utils.gen_pict.gen_pict_use_one_fun(reconstructed_X(i).map(x=>math.round(x*255.0).toInt).toArray, 28, 28,"D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/test/da_mnist_test/dA_mnist_test_"+i+"pred.jpg")
      writer_test_out.write("\n")
    }
    writer_test_out.close()
  }

  def main(args: Array[String]) {
    //test_dA_simple()
    //dA_train_mnist();
    dA_test_mnist();
  }
  
}
