package dp_process_parallel
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

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
    val filePath_tain:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/dataset_face/hdfs/lfw_5590_train_min.txt"
    val train_X=dp_utils.dataset.readtxt(filePath_tain).map(x=>dp_utils.gen_rgb.split_rgb_line(x))
    
    val females=500
    val males=500
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