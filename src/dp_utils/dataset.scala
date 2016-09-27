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
    var result:ArrayBuffer[Tuple2[Int,Array[Double]]]=ArrayBuffer();
    val read_datas=readtxt(filePath)
    read_datas.foreach(line=>{val line_split:Array[String]=line.split("=sep=",2);result+=Tuple2(Integer.parseInt(line_split(1)),line_split(0).split(",").map(x=>x.toDouble))})
    result.toArray
  }
  
  def main(args:Array[String]):Unit={
    val filepath_mnist_valid="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/valid_data.txt"
    var demo=readtxt(filepath_mnist_valid)
    println(demo(0))
    println(demo(0).map(x=>x.toDouble)reduce(_+_))
    var demo2=load_mnist(filepath_mnist_valid)
    println("lable="+demo2(0)._1+"\t piex="+demo2(0)._2.mkString(","))
  }
}