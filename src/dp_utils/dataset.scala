package dp_utils

import scala.io.Source
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array

  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

object dataset {
  /*vision1
   * def readtxt(filePath:String):Array[String]={
    var result:ArrayBuffer[String]=ArrayBuffer();
    Source.fromFile(filePath).getLines().foreach(line=>result+=line) 
    result.toArray
  }*/
  def readtxt(filePath:String):Array[String]={
    Source.fromFile(filePath).getLines().toArray 
  }
  
  //每一行表示一个28x28像素的0-9手写图片,像素值0-1之间
  //存储方式为tuple2(0-9,Array[0.0,0.1,.....0.19,0.93,......0.0,0.0])
  def load_mnist(filePath:String):Array[Tuple2[Int,Array[Double]]]={
    val result:ArrayBuffer[Tuple2[Int,Array[Double]]]=ArrayBuffer();
    val read_datas=readtxt(filePath)
    read_datas.foreach(line=>{val line_split:Array[String]=line.split("=sep=",2);result+=Tuple2(Integer.parseInt(line_split(1)),line_split(0).split(",").map(x=>x.toDouble))})
    result.toArray
  }
  
  /* 读取code_filePath内的lexs,nes,labels
   * 读取 dict_words_filePath内的words
   * 读取dict_labels_filePath内的labels
   */
  def load_fold(code_filePath:String,dict_words_filePath:String,dict_labels_filePath:String):(Array[Array[Int]],Array[Array[Int]],Array[Array[Int]],Map[Int,String],Map[Int,String])={
    val code_data=readtxt(code_filePath).map(x=>x.split("=sep2="))
    val len_code_data=code_data.length
    val lexs:ArrayBuffer[Array[Int]]=ArrayBuffer()
    val nes:ArrayBuffer[Array[Int]]=ArrayBuffer()
    val labels:ArrayBuffer[Array[Int]]=ArrayBuffer()
    for(i <- 0 until len_code_data){
     lexs   += code_data(i)(0).split("=sep1=").map(x=>x.toInt)
     nes    += code_data(i)(1).split("=sep1=").map(x=>x.toInt)
     labels += code_data(i)(2).split("=sep1=").map(x=>x.toInt)
    }
    val code2words:Map[Int,String]=readtxt(dict_words_filePath).map(x=>{val tmp=x.split("=sep1=");(tmp(1).toInt->tmp(0))}).toMap
    val code2labels:Map[Int,String]=readtxt(dict_labels_filePath).map(x=>{val tmp=x.split("=sep1=");(tmp(1).toInt->tmp(0))}).toMap
    (lexs.toArray,nes.toArray,labels.toArray,code2words,code2labels)
  }
  
  def main(args:Array[String]):Unit={
    val filepath_mnist_valid="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/valid_data.txt"
    var demo=readtxt(filepath_mnist_valid)
    println(demo(0))
    println(demo(0).map(x=>x.toDouble)reduce(_+_))
    var demo2=load_mnist(filepath_mnist_valid)
    println("lable="+demo2(0)._1+"\t piex="+demo2(0)._2.mkString(","))
    var demo3=load_fold("D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/fold/fold_code.txt",
                        "D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/fold/fold_dict_words.txt",
                        "D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/fold/fold_dict_labels.txt")
    print(demo3._1.length+"\t"+demo3._2.length+"\t"+demo3._3.length+"\t"+demo3._4.size+"\t"+demo3._5.size)
  }
}