package dp_utils

//由jpg解析rgb  以及生产hdfs数据 
import java.awt.AWTException;  
import java.awt.Dimension;  
import java.awt.Rectangle;  
import java.awt.Robot;  
import java.awt.Toolkit;  
import java.awt.image.BufferedImage;  
import java.io.File;  
import java.io._
import javax.imageio.ImageIO;  

//scala
import scala.collection.mutable.ArrayBuffer    //用于建立可变的array


  /**
   * deeplearning with scala and spark
   *
   * Copyright liming(oldlee11)
   * Email: oldlee11@163.com
   * qq:568677413
   */

//一像素的rgb数据
class pix_rgb_struct(i_in:Int,j_in:Int,rgb_in:Array[Int]){
  var i:Int=i_in;//像素的行位置
  var j:Int=j_in;//像素的列位置
  var rgb:Array[Int]=rgb_in;
}


//一个图片的rgb数据
class pict_rgb_struct(width_in:Int,heigth_in:Int,rgb_values_in:Array[pix_rgb_struct]){
  var width:Int=width_in;
  var heigth:Int=heigth_in;
  var rgb_values:Array[pix_rgb_struct]=rgb_values_in;
}



object gen_rgb {
    /* 方案1:在本机处理图片   
    //jpg->rbm
    def getImagePixel(image:String):String={  
      try {        
        var rgb:List[Int] = List(0,0,0);  
        val file:File = new File(image);  
        val bi:BufferedImage = ImageIO.read(file);  
        val width:Int= bi.getWidth();  
        val height:Int = bi.getHeight();  
        val minx:Int = bi.getMinX();  
        val miny:Int = bi.getMinY();  
        var result:String=width+"=sep1="+height+"=sep1=";
        var RGB_result:ArrayBuffer[String]=ArrayBuffer();
        var tmp_rgb_i:String="";
        /*
        println(width)
        println(height)
        */
        for (i:Int <- minx to (width-1)) {  
            for (j:Int <- miny to (height-1)) {
                val pixel = bi.getRGB(i,j);           
                rgb= List((pixel & 0xff0000) >> 16, (pixel & 0xff00) >> 8,pixel & 0xff)
                /* debug
                println(i)
                println(j)
                println(pixel)
                println((pixel & 0xff0000) >> 16)
                println((pixel & 0xff00) >> 8)
                println(pixel & 0xff)                
                println(i)
                println(j)
                println(rgb)                
                */                     
                tmp_rgb_i=
                  if((i==width-1) && (j==height-1)){
                    i+"=sep3="+j+"=sep3="+rgb(0)+"=sep4="+rgb(1)+"=sep4="+rgb(2)+"";	
                  }else{
                    i+"=sep3="+j+"=sep3="+rgb(0)+"=sep4="+rgb(1)+"=sep4="+rgb(2)+"=sep2=";	
                  }
                RGB_result += tmp_rgb_i;
            }  
        } 
        result+RGB_result.mkString
      } catch {
		    case ex: Exception => "-1"
		  }        
    }  
    
    //str_in="533=sep1=800=sep1=0=sep3=0=sep3=0=sep4=123=sep4=191=sep2=0=sep3=1=sep3=0=sep4=122=sep4=190=sep2=0=se........"
    def split_rgb_line(str_in:String):pict_rgb_struct={
      val str_in_split1:Array[String]=str_in.split("=sep1=")
      val width=Integer.parseInt(str_in_split1(0))
      val heigth=Integer.parseInt(str_in_split1(1))
      val str_in_split2:Array[String]=str_in_split1(2).split("=sep2=")
      val tmp:Array[String] =str_in_split2(0).split("=sep3=");
      val tmp2:pix_rgb_struct=new pix_rgb_struct(Integer.parseInt(tmp(0)),Integer.parseInt(tmp(1)),tmp(2).split("=sep4=").map(y=>Integer.parseInt(y)))
      val rgb_valuse:Array[pix_rgb_struct]=str_in_split2.map(x=>{val tmp:Array[String] =x.split("=sep3=");new pix_rgb_struct(Integer.parseInt(tmp(0)),Integer.parseInt(tmp(1)),tmp(2).split("=sep4=").map(y=>Integer.parseInt(y)))})
      new pict_rgb_struct(width,heigth,rgb_valuse)
    }*/

  
    /*方案2 在集群处理图片*/
    //image=本机图片的地址
    //输出 宽+高+没有处理的pixel数据组成的字符串  由行到列排列
    //533,800=sep=-16745537,-16745794,-16746051,-16746565,-16747335,-16748105,-16748364,........
    def getImagePixel(image:String):String={ 
      try {        
        val file:File = new File(image);  
        val bi:BufferedImage = ImageIO.read(file);  
        val width:Int= bi.getWidth();  
        val height:Int = bi.getHeight();  
        val minx:Int = bi.getMinX();  
        val miny:Int = bi.getMinY();  
        var RGB_result:ArrayBuffer[Int]=ArrayBuffer();
        var tmp_rgb_i:String="";
        for (i:Int <- minx to (width-1)) {  
            for (j:Int <- miny to (height-1)) {
                val pixel = bi.getRGB(i,j);
                RGB_result += pixel;
            }  
        } 
        width+","+height+"=sep="+RGB_result.mkString(sep=",")
      } catch {
		    case ex: Exception => "-1"
		  }              
    }
    
    //str_in="533,800=sep=-16745537,-16745794,-16746051,-16746565,-16747335,-16748105,-16748364,........"
    //输出=一个 pict_rgb_struct数据结构
    def split_rgb_line(str_in:String):pict_rgb_struct={
      val str_in_split1:Array[String]=str_in.split("=sep=")
      val width=Integer.parseInt(str_in_split1(0).split(",")(0))
      val heigth=Integer.parseInt(str_in_split1(0).split(",")(1))
      val str_in_split2:Array[String]=str_in_split1(1).split(",")
      val rgb_valuse:ArrayBuffer[pix_rgb_struct]=ArrayBuffer()
      for(i <-0 until width){
        for(j<- 0 until heigth){
          val pixel=str_in_split2(i*width+j).toInt
          val rgb_array=Array((pixel & 0xff0000) >> 16, (pixel & 0xff00) >> 8,pixel & 0xff)
          val rgb_valuse_i_j:pix_rgb_struct=new pix_rgb_struct(i_in=i,j_in=j,rgb_in=rgb_array) 
          rgb_valuse += rgb_valuse_i_j
        }
      }
      new pict_rgb_struct(width,heigth,rgb_valuse.toArray)
    }  
    
    //strin-->array  
    def split_rgb_line2(str_in:String):Array[Array[Array[Double]]]={
      val str_in_split1:Array[String]=str_in.split("\t")(1).split("=sep=")
      val width=Integer.parseInt(str_in_split1(0).split(",")(0))
      val heigth=Integer.parseInt(str_in_split1(0).split(",")(1))
      val result:Array[Array[Array[Double]]]=Array.ofDim[Double](3,heigth,width)
      val str_in_split2:Array[String]=str_in_split1(1).split(",")
      for (j:Int <- 0 to (heigth-1)) {
        for (i:Int <- 0 to (width-1)) {  
          val pixel = str_in_split2(j*width+i).toInt;           
          val rgb_array=Array((pixel & 0xff0000) >> 16, (pixel & 0xff00) >> 8,pixel & 0xff)
          result(0)(j)(i)=rgb_array(0).toDouble/255.0
          result(1)(j)(i)=rgb_array(1).toDouble/255.0
          result(2)(j)(i)=rgb_array(2).toDouble/255.0
        }  
      } 
      result
    }  
    
        
    //给定一个目录 
    //返回里面文件的列表
    def subdirs2(dir: File): Iterator[File] = { 
		  val d = dir.listFiles.filter(_.isDirectory)
		  val f = dir.listFiles.filter(_.isFile).toIterator
		  f ++ d.toIterator.flatMap(subdirs2)
	  } 
    
    //把jpg数据解析为rbm并生产hdfs使用的txt文件
    def gen_hdfs_txt(dir:String,out_file_path:String)={
      val jpgs=subdirs2(new File(dir))
      val writer = new PrintWriter(new File(out_file_path))
      jpgs.foreach(x => {val jpg_full_name=x.toPath().toString();val jpg_name_tmp=jpg_full_name.split("//");val jpg_name=jpg_name_tmp(jpg_name_tmp.length-1);writer.write(jpg_name+"\t"+getImagePixel(jpg_full_name));writer.write("\n");println(jpg_name)})
      writer.close()         
    }
  
    /** 
     * @param args 
     */  
    def main(args:Array[String]):Unit={ 

        /*
         * test for getImagePixel   jpg转rgb
         * */        
        var demo:String=getImagePixel("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//test//001.jpg"); 
        println(demo.substring(0, 200)) 
        //rgb 取值为 0-255
        //width    height         行                    列                    rgb                            行                   列           rgb            ........
        //533=sep1=800=sep1=0=sep3=0=sep3=0=sep4=123=sep4=191=sep2=0=sep3=1=sep3=0=sep4=122=sep4=190=sep2=0=sep3=2=sep3=0=sep4=121=sep4=189=sep2=0=sep3=3=sep3=0=sep4=119=sep4=187=sep2=0=sep3=4=sep3=0=sep4=116=s

        
        /*
         * test for split_rgb_line   测试解析getImagePixel输出的文本数据  用于hdfs读取
         * */       
        val demo2=split_rgb_line(demo)
        println("width="+demo2.width)
        println("height="+demo2.heigth)
        println("rgb:")
        var rgb_tmp= new pix_rgb_struct(0,0,Array())
        for(i <-0 to 10){
          rgb_tmp=demo2.rgb_values(i);
          println("i="+rgb_tmp.i+",j="+rgb_tmp.j+",rgb=("+rgb_tmp.rgb.mkString(sep=",")+")")
        }
        /*width=533
          height=800
          rgb:
          i=0,j=0,rgb=(0,123,191)
          i=0,j=1,rgb=(0,122,190)
          i=0,j=2,rgb=(0,121,189)
          .........
         */
        
        /*
         * test for gen_hdfs_txt   用于在本地把jpg图片转化为rgb形式的数据字符串(每个图片一行数据),用于存入hdfs内  
         * 产生数据很大
         * */ 
        //gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590.txt")
        //gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//net_7876","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//net_7876.txt")         
        gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//train//female","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590_female_train.txt") 
        gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//train//male","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590_male_train.txt") 
        gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//test//female","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590_female_test.txt") 
        gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//test//male","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590_male_test.txt")         

    }    
}