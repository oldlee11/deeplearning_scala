package dp_process_parallel

  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

object test {
  
  def train_test_face_data(){
    val width:Int=250;
    val height:Int=250;
    /*
     * 
     * train
     * 
     * */
    var filePath_female:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/dataset_face/hdfs/lfw_5590_female_train.txt"
    var filePath_male:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/dataset_face/hdfs/lfw_5590_male_train.txt"
    /*
    val read_datas_female=dp_utils.dataset.readtxt(filePath_female)
    val read_datas_male=dp_utils.dataset.readtxt(filePath_male)
    var train_female_x:Array[Array[Array[Array[Double]]]]=read_datas_female.map(x=>dp_utils.gen_rgb.split_rgb_line2(x))
    var train_male_x:Array[Array[Array[Array[Double]]]]=read_datas_male.map(x=>dp_utils.gen_rgb.split_rgb_line2(x))
    var train_female_y:Array[Int]=new Array(train_female_x.length)
    var train_male_y:Array[Int]=new Array(train_male_x.length)
    for(i <-0 until train_male_x.length){
      train_male_y(i)=1
    }
    val train_X=train_female_x++train_male_x
    val train_Y=(train_female_y++train_male_y).map(x=>Array(x))
    val train_N: Int = train_X.length
    */
    val females=975
    val train_X=dp_utils.dataset.readtxt(filePath_female).map(x=>dp_utils.gen_rgb.split_rgb_line2(x)) ++ dp_utils.dataset.readtxt(filePath_male).map(x=>dp_utils.gen_rgb.split_rgb_line2(x))
    val train_Y_init:Array[Int]=new Array(train_X.length)
    for(i <-0 until females){
      train_Y_init(i)=1
    }   
    val train_Y=(train_Y_init).map(x=>Array(x))
    val train_N: Int = train_X.length
    
    var learning_rate: Double = 0.1
    val n_epochs: Int = 100
    //lenet5
    val classifier = new  CNN_parallel(input_size=(height,width),output_size=1,n_kernel_Array=Array(16,26,56,100,120),kernel_size_Array=Array((25,25),(10,10),(9,9),(8,8),(2,2)),pool_size_Array=Array((2,2),(2,2),(4,4),(2,2),(1,1)),n_channel=1,n_hidden=84,_rng=null,activation="ReLU",activation_mlp="tanh")
 
    // train
    var epoch: Int = 0
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train_batch(inputs_x=train_X, inputs_y=train_Y, lr=learning_rate, batch_num_per=0.1,alpha=0.0)
      //learning_rate *=0.99
    } 
  }
  
  def main(args: Array[String]) {
    train_test_face_data()
  }    
}