package dp_process

// $ scalac RBM.scala
// $ scala RBM

import scala.util.Random
import scala.math
import scala.math.log

import java.io.File; 
import java.io._
import scala.io.Source

import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

/*
 * n_visible 可视层单元个数
 * n_hidden  隐藏层单元个数
 * _W 可视层和隐层层之间的权重系数 默认为null空
 * _hbias:隐层层的偏置,默认为null空
 * _vbias:可视层的偏置,默认为null空
 * rng:随机种子,默认为null空
 * */
class RBM(val n_visible: Int, val n_hidden: Int,
          _W: Array[Array[Double]]=null, _hbias: Array[Double]=null, _vbias: Array[Double]=null,
          var rng: Random=null) {
  
  var W: Array[Array[Double]] = Array.ofDim[Double](n_hidden, n_visible)//隐藏层和可视层间的权重系数 w权重是 n_hidden行  x n_visible列的 Double 型2维矩阵
  var hbias: Array[Double] = new Array[Double](n_hidden)//每个隐藏层单元的偏置 hbias是 n_hidden长度的Double 型1维向量
  var vbias: Array[Double] = new Array[Double](n_visible)//每个可视层单元的偏置 n_visible是 n_hidden长度的Double 型1维向量
  var cross_entropy_result:Double=0.0
  
  if(rng == null) rng = new Random(1234)//如果没有输入随机种子,则设置为Random(1234)

  init_w_module()
  
  if(_W == null) {
  } else {
    //如果_W不为空则使用_W初始化W
    W = _W
  }

  if(_hbias == null) {
  } else {
    hbias = _hbias
  }

  if(_vbias == null) {
  } else {
    vbias = _vbias
  }

  //参见RBM
  //定义uniform均值分布的随机数产生函数
  //每次仅产生一个随机数  
  def uniform(min: Double, max: Double): Double = rng.nextDouble() * (max - min) + min
  
  //参见RBM
  //定义二项分布的随机数产生函数
  //n是二项分布数据的个数,例如=100个
  //p是二项分=1的概率值,例如=0.2
  //返回值=n中取值=1的个数约等于 =100*0.2=20 ,随机值
  //如果n=1则为产生一个二项分布的随机数  
  def binomial(n: Int, p: Double): Int = {
    if(p < 0 || p > 1) return 0
    var c: Int = 0
    var r: Double = 0
    var i: Int = 0
    for(i <- 0 until n) {
      r = rng.nextDouble()
      if(r < p) c += 1
    }
    c
  }

  //参见RBM
  //非线性映射函数=1/(1+E(-x次幂))
  def sigmoid(x: Double): Double = 1.0 / (1.0 + math.pow(math.E, -x))

  
  //使用contrastive_divergence做批量训练
  //batch_num_per  从样本集合inputs中随机抽取一些样本,抽取的个数为样本总数*batch_num_per
  def contrastive_divergence_batch(inputs: Array[Array[Double]], lr: Double, k: Int, batch_num_per:Double):Unit={
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
      contrastive_divergence(inputs(i), lr, k,batch_num)
    }
    //完成一批次训练后计算本批次的平均交叉嫡cross_entropy_result/batch_num (cross_entropy_result内部是累加的)
    println("cross_entropy="+cross_entropy_result/batch_num)      
  }
  
  
  /*输入一个样本,并训练更新一下系数(更新时要除以batch_num)
    lr是参数的学习率默认0.1
    k是反射抽样h和v的反射次数(最终会用于估算参数梯度,一般k=1就可以估算了)
    input 是一个可视层的输入样本
              最后完成一次对w参数的迭代  
    batch_num 是一次批量迭代使用的样本数,由于每一次训练时,样本仅输入1个,参数(W,vbias,hbias)迭代计算时：W(t+1)=W(t)+lr*(局部误差/batch_num)
                                                 即每一次迭代仅仅更新1/batch_num倍,如果batch_num=训练集样本个数,则当所有训练集都迭代一遍后则 为 W(t+1)=W(t)+lr*(局部误差)
                                                如果batch_num=1则表示每个样本迭代一次 W(t+1)=W(t)+lr*(局部误差) 即不是批量迭代而是随机迭代了
  */
  def contrastive_divergence(input: Array[Double], lr: Double, k: Int, batch_num:Int) {
    val ph_mean: Array[Double] = new Array[Double](n_hidden)
    val ph_sample: Array[Int] = new Array[Int](n_hidden)
    val nv_means: Array[Double] = new Array[Double](n_visible)
    val nv_samples: Array[Double] = new Array[Double](n_visible)
    val nh_means: Array[Double] = new Array[Double](n_hidden)
    val nh_samples: Array[Int] = new Array[Int](n_hidden)

    /* CD-k */
    //由一个可视层样本数据input,抽样出一个隐藏层样本数据以及概率
    sample_h_given_v(input, ph_mean, ph_sample)
    var step: Int = 0
    //遍历反射次数k(经过k次反射),抽样出k次后的隐藏层样本数据和可视层样本数据以及概率
    for(step <- 0 until k) {
      if(step == 0) {
        gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples)
      } else {
        gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples)
      }
    }

    var input_reconstruct:Array[Double]=new Array(input.length)
    reconstruct(input, input_reconstruct)//计算编解码后的数据input_reconstruct
    this.cross_entropy_result=this.cross_entropy_result + cross_entropy(input,input_reconstruct)//衡量输入和经过编解码后的输出数据之间的相似度(使用KL散度,即相对信息嫡)
    
    //更新w vbias hbias
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_hidden) {
      for(j <- 0 until n_visible) {
        // W(i)(j) += lr * (ph_sample(i) * input(j) - nh_means(i) * nv_samples(j)) / batch_num
        //更新w
        W(i)(j) += lr * (ph_mean(i) * input(j) - nh_means(i) * nv_samples(j)) / batch_num
      }
      //更新hbias
      hbias(i) += lr * (ph_sample(i) - nh_means(i)) / batch_num
    }
    for(i <- 0 until n_visible) {
      //更新vbias
      vbias(i) += lr * (input(i) - nv_samples(i)) / batch_num
    }
  }

  //根据一个可视层v样本和p(h|v)抽样得到一个隐藏层h样本(样本只能=0或1)
  def sample_h_given_v(v0_sample: Array[Double], mean: Array[Double], sample: Array[Int]) {
    var i: Int = 0
    for(i <- 0 until n_hidden) {
      mean(i) = propup(v0_sample, W(i), hbias(i))//计算p(h|v),即获得了隐藏层中每个单元取1的概率
      sample(i) = binomial(1, mean(i))//根据每个隐藏层单元取1的概率h1_mean,和二项分,随机产生一个隐藏层数据:h1
    }
  }

  //根据一个隐藏层h样本和p(v|h)抽样得到一个可视层v样本,(样本只能=0或1)
  def sample_v_given_h(h0_sample: Array[Int], mean: Array[Double], sample: Array[Double]) {
    var i: Int = 0
    for(i <- 0 until n_visible) {
      mean(i) = propdown(h0_sample, i, vbias(i))//计算p(v|h),即获得了可视层中每个单元取1的概率
      sample(i) = binomial(1, mean(i)).toDouble//根据每个可视层单元取1的概率v1_mean,和二项分,随机产生一个可视层数据:v1
    }
  }

  /*
        计算p(h|v),即在可视层单元=v时,获得隐藏层中每个单元取1的概率
        把可视层样本数据v和w加权求和并加hbias
        然后做simgoid映射处理 
   * */
  def propup(v: Array[Double], w: Array[Double], b: Double): Double = {
    var pre_sigmoid_activation: Double = 0
    var j: Int = 0
    for(j <- 0 until n_visible) {
      pre_sigmoid_activation += w(j) * v(j)
    }
    pre_sigmoid_activation += b
    sigmoid(pre_sigmoid_activation)
  }

  /*
        计算p(v|h),即在隐藏层单元=h时,获得可视层中每个单元取1的概率
        把隐藏层样本数据h和w加权求和并加vbias
        然后做simgoid映射处理
   * */
  def propdown(h: Array[Int], i: Int, b: Double): Double = {
    var pre_sigmoid_activation: Double = 0
    var j: Int = 0
    for(j <- 0 until n_hidden) {
      pre_sigmoid_activation += W(j)(i) * h(j)
    }
    pre_sigmoid_activation += b
    sigmoid(pre_sigmoid_activation)
  }

  /*
         输入一个隐藏层的样本,经过一次反射后产生的可视层和隐藏层样本
   h-->v-->h
   * */
  def gibbs_hvh(h0_sample: Array[Int], nv_means: Array[Double], nv_samples: Array[Double], nh_means: Array[Double], nh_samples: Array[Int]) {
    sample_v_given_h(h0_sample, nv_means, nv_samples)
    sample_h_given_v(nv_samples, nh_means, nh_samples)
  }

  //经过rbm编解码的数据
  def reconstruct(v: Array[Double], reconstructed_v: Array[Double]) {
    val h: Array[Double] = new Array[Double](n_hidden)
    var pre_sigmoid_activation: Double = 0
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_hidden) {
      h(i) = propup(v, W(i), hbias(i))
    }
    for(i <- 0 until n_visible) {
      pre_sigmoid_activation = 0
      for(j <- 0 until n_hidden) {
        pre_sigmoid_activation += W(j)(i) * h(j)
      }
      pre_sigmoid_activation += vbias(i)
      reconstructed_v(i) = sigmoid(pre_sigmoid_activation)
    }
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
  //并修改RBM的W    vbias    hbias
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


object RBM {
  def test_RBM_simple() {
    val rng: Random = new Random(123)
    var learning_rate: Double = 0.1
    val training_epochs: Int = 1000
    val k: Int = 1
    val train_N: Int = 6;
    val test_N: Int = 2
    val n_visible: Int = 6
    val n_hidden: Int = 3
    val train_X: Array[Array[Double]] = Array(
      Array(1, 1, 1, 0, 0, 0),
      Array(1, 0, 1, 0, 0, 0),
      Array(1, 1, 1, 0, 0, 0),
      Array(0, 0, 1, 1, 1, 0),
      Array(0, 0, 1, 0, 1, 0),
      Array(0, 0, 1, 1, 1, 0)
    )
    val rbm: RBM = new RBM(n_visible, n_hidden, rng=rng)
    var i: Int = 0
    var j: Int = 0
    // train
    var epoch: Int = 0
    for(epoch <- 0 until training_epochs) {
      /*for(i <- 0 until train_N) {
        rbm.contrastive_divergence(train_X(i), learning_rate, k,train_N)
      }*/
      rbm.contrastive_divergence_batch(train_X, learning_rate, k,1.0)
    }
    // test data
    val test_X: Array[Array[Double]] = Array(
      Array(1, 1, 0, 0, 0, 0),
      Array(0, 0, 0, 1, 1, 0)
    )
    val reconstructed_X: Array[Array[Double]] = Array.ofDim[Double](test_N, n_visible)
    for(i <- 0 until test_N) {
      rbm.reconstruct(test_X(i), reconstructed_X(i))
      for(j <- 0 until n_visible) {
        printf("%.5f ", reconstructed_X(i)(j))
      }
      println()
    }
    //保存系数
    rbm.save_w("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//rbm_simple_module.txt")
    //观看rbm系数的图像
    rbm.see_train_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//",3,2)
    //清空系数
    rbm.init_w_module()
    //读取系数
    rbm.read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//rbm_simple_module.txt")
    //再次保存系数
    rbm.save_w("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//rbm_simple_module2.txt")
  }

  //使用mnist数据集进行训练
  def RBM_train_mnist():Unit={
    //读取训练集数据
    val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/training_data.txt"
    val train_X:Array[Array[Double]]=dp_utils.dataset.load_mnist(filePath_train).map(x=>x._2)
    val train_N: Int = train_X.length
    
    //设置rbm的基础参数并建立rbm
    val rng: Random = new Random(123)
    var learning_rate: Double = 0.1//学习率,步长
    val k: Int = 1
    val training_epochs: Int = 200//批量迭代次数=500
    val n_visible: Int = 28 * 28//mnist是28X28的像素图片,所以每个可视层的单元对应一个位置的像素值
    val n_hidden: Int = 100//隐藏层500个单元=提取500个特征    
    //由于训练集太多,这里每次批量的个数<>train_N 而是随机抽取一定比例的样本,比例=batch_num_per
    val batch_num_per:Double=0.3  //每次批量训练 提取训练样本的比例   =1.0表示不抽样   =20%表示每次抽样20%的样本做一次批量训练
    val rbm: RBM = new RBM(n_visible, n_hidden, rng=rng)
    
    // train
    var epoch: Int = 0
    //批量训练 training_epochs次
    for(epoch <- 0 until training_epochs) {
      println("第"+epoch+"次迭代:")
      rbm.contrastive_divergence_batch(train_X, learning_rate, k,batch_num_per)
      if(epoch==(training_epochs-1) || epoch==math.round(training_epochs.toDouble/4.0) || epoch==math.round(2.0*training_epochs.toDouble/4.0) || epoch==math.round(3.0*training_epochs.toDouble/4.0)){
        //记录权值w
        rbm.save_w("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//RBM_mnist_module"+epoch+".txt")
      }
    }  
    //观看rbm系数的图像
    rbm.see_train_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//",28,28)
  }  
  
 //使用mnist数据集进行测试
  def RBM_test_mnist():Unit={
    //载入测试集数据
    val filePath_test:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt"
    val test_X:Array[Array[Double]]=dp_utils.dataset.load_mnist(filePath_test).map(x=>x._2)
    val test_N: Int = test_X.length
    
    //设置rbm的基础参数并建立rbm
    val n_visible: Int = 28 * 28//mnist是28X28的像素图片,所以每个可视层的单元对应一个位置的像素值
    val n_hidden: Int = 100//隐藏层500个单元=提取500个特征        
    val rng: Random = new Random(123)
    val rbm: RBM = new RBM(n_visible, n_hidden, rng=rng)
    //使用训练时生成的权值,初始化测试rbm的权值w
    rbm.read_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//RBM_mnist_module100.txt")
    rbm.see_train_w_module("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//module_out//",28,28)
    
    //test
    val writer_test_out = new PrintWriter(new File("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//test//RBM_mnist_test.txt"))
    val reconstructed_X: Array[Array[Double]] = Array.ofDim[Double](test_N, n_visible)
    for(i <- 0 until test_N) {
      rbm.reconstruct(test_X(i), reconstructed_X(i))
      writer_test_out.write("样本"+i+":\n")
      writer_test_out.write("    输入数据=")
      for(j <- 0 until n_visible) {
        writer_test_out.write(test_X(i)(j).toString()+",")
      }
      //生成图片
      dp_utils.gen_pict.gen_pict_use_one_fun(test_X(i).map(x=>math.round(x*255.0).toInt).toArray, 28, 28,"D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/test/RBM_mnist_test/RBM_mnist_test_"+i+".jpg")
      writer_test_out.write("\n")
      writer_test_out.write("    预测数据=")      
      for(j <- 0 until n_visible) {
        writer_test_out.write(reconstructed_X(i)(j).toString()+",")
      }
      //生成图片
      dp_utils.gen_pict.gen_pict_use_one_fun(reconstructed_X(i).map(x=>math.round(x*255.0).toInt).toArray, 28, 28,"D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/test/RBM_mnist_test/RBM_mnist_test_"+i+"pred.jpg")
      writer_test_out.write("\n")
    }
    writer_test_out.close()
  }
  
  def main(args: Array[String]) {
    test_RBM_simple()
    /*
0.89614 0.61554 0.98937 0.06006 0.09354 0.00802 
0.07076 0.04888 0.98222 0.69151 0.91911 0.01201 
     * */
    //RBM_train_mnist()
    //RBM_test_mnist()
  }

}
