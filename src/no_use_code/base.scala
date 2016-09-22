package dp_process_parallel

//用于并行加速运算
object base {

  //并行计算Array_in内所有元素的总和
  def parallel_sum(Array_in:Array[Double],type_in:String="1"):Double={
    val error:Double=0.0
    if(type_in=="1"){
      Array_in.par.fold(0.0)((x,y)=>x+y)
    }else{
      error
    }
  }  
  
  //并行计算Array1_in和Array2_in内每个元素相乘
  //Array(1.0,0.91,2.1)
  //       *   *    *
  //Array(2.0,0.0,3.0)
  //       ||  ||  ||
  //Array(2.0,0.0,6.3)
  def parallel_mult2(Array1_in:Array[Double],Array2_in:Array[Double],type_in:String="1"):Array[Double]={
    val error:Array[Double]=Array()
    if(Array1_in.length==Array2_in.length){
      if(type_in=="1"){
        Array1_in.par.zip(Array2_in.par).map(x=>x._1*x._2).toArray  //vision1
        //Array1_in.zip(Array2_in).par.map(x=>x._1*x._2).toArray    //vision2
      }else{
        error
      }  
    }else{
      error
    }    
  }

  //并行计算Array1_in和Array2_in内每个元素相乘,然后相加
  //Array(1.0,0.91,2.1)
  //       *   *    *
  //Array(2.0,0.0,3.0)
  //       ||  ||  ||
  //Array(2.0,0.0,6.3)  
  //         sum
  //         8.3
  def parallel_mult2_sum(Array1_in:Array[Double],Array2_in:Array[Double],type_in:String="1"):Double={
    val error:Double=0.0
    if(type_in=="1"){
      Array1_in.par.zip(Array2_in.par).map(x=>x._1*x._2).fold(0.0)((x,y)=>x+y)  //vision1
      //Array1_in.zip(Array2_in).par.map(x=>x._1*x._2).fold(0.0)((x,y)=>x+y)    //vision2
      
    }else{
      0.0
    }   
  }

  /*
   * Array(Array(1.0,0.91,2.1),
   *       Array(1.0,0.91,2.1))
   *        +
   * Array(Array(2.0,0.0,3.0),
   *       Array(2.0,0.0,3.0))
   *       =
   * Array(Array(3.0,0.91,5.1),
   *       Array(3.0,0.91,5.1))  
   * */
  def parallel_2d_add2(Array2d_in1:Array[Array[Double]],Array2d_in2:Array[Array[Double]],type_in:String="1"):Array[Array[Double]]={
    val error:Array[Array[Double]]=Array()
    if(type_in=="1"){
      (0 until Array2d_in1.length).par.map(i=>(0 until Array2d_in1(0).length).par.map(j=>Array2d_in1(i)(j)+Array2d_in2(i)(j)).toArray).toArray
    }else{
      error
    }
  }
  
  def main(args:Array[String]):Unit={ 
    //test parallel_sum
    var test_vector:Array[Int]=(1 to 10000000).toArray
    var result=parallel_sum(test_vector.map(x=>x.toDouble))
    println("test:parallel_sum="+result)
    //test parallel_mult2
    var test_vector1:Array[Int]=(1 to 10000000).toArray
    var test_vector2:Array[Int]=(2 to 10000001).toArray
    var result2:Array[Double]=parallel_mult2(test_vector1.map(x=>x.toDouble),test_vector2.map(x=>x.toDouble))
    println("parallel_mult2="+result2)   
    //test parallel_mult2_sum
    result=parallel_mult2_sum(test_vector1.map(x=>x.toDouble),test_vector2.map(x=>x.toDouble))
    println("parallel_mult2_sum="+result)   
  }    
}