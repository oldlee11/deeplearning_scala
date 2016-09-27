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
    //width,hitght=sep=rgb.......
    //533,800=sep=-16745537,-16745794,-16746051,-16746565,-16747335,-16748105,-16748364,........
    def getImagePixel(image:String):String={ 
      try {        
        val file:File = new File(image);  
        val bi:BufferedImage = ImageIO.read(file);  
        val width:Int= bi.getWidth();  
        val height:Int = bi.getHeight();  
        val minx:Int = bi.getMinX();  
        val miny:Int = bi.getMinY();  
        var RGB_result:Array[Int]=new Array((width-minx)*(height-miny));
        var tmp_rgb_i:String="";
        for (i:Int <- minx until width) {  
            for (j:Int <- miny until height) {
                RGB_result(j*(width-minx)+i) = bi.getRGB(i,j);
            }  
        } 
        width+","+height+"=sep="+RGB_result.mkString(sep=",")
      } catch {
		    case ex: Exception => "-1"
		  }              
    }
    
    //str_in="533,800=sep=-16745537,-16745794,-16746051,-16746565,-16747335,-16748105,-16748364,........" 
    //strin-->array  
    /*输出=Array[//第几个channel
            Array[//像素位置:第几行
              Array[Double]//像素位置:第几列
            ]
          ]
    * 
    */
    def split_rgb_line(str_in:String,max_value:Double=255.0):Array[Array[Array[Double]]]={
      val str_in_split1:Array[String]=str_in.split("=sep=")
      val width=str_in_split1(0).split(",")(0).toInt
      val heigth=str_in_split1(0).split(",")(1).toInt
      val result:Array[Array[Array[Double]]]=Array.ofDim[Double](3,heigth,width)
      val str_in_split2:Array[String]=str_in_split1(1).split(",")
      for(j<- 0 until heigth){ 
        for(i <-0 until width){
          val pixel = str_in_split2(j*width+i).toInt;           
          val rgb_array=Array((pixel & 0xff0000) >> 16, (pixel & 0xff00) >> 8,pixel & 0xff)
          result(0)(j)(i)=rgb_array(0).toDouble/max_value
          result(1)(j)(i)=rgb_array(1).toDouble/max_value
          result(2)(j)(i)=rgb_array(2).toDouble/max_value
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
    
    //把一个目录下的多个jpg文件解析为rbm并生产hdfs使用的txt文件
    //533,800=sep=-16745537,-16745794,-16746051,-16746565,-16747335,-16748105,-16........
    def gen_hdfs_txt(dir:String,out_file_path:String)={
      val jpgs=subdirs2(new File(dir))
      val writer = new PrintWriter(new File(out_file_path))
      jpgs.foreach(x => {val jpg_full_name=x.toPath().toString();writer.write(getImagePixel(jpg_full_name));writer.write("\n");println(jpg_full_name)})
      writer.close()         
    }
    //把多个目录下的多个jpg文件解析为rbm并生产hdfs使用的txt文件
    def gen_hdfs_txts(dirs:Array[String],out_file_path:String)={
      val writer = new PrintWriter(new File(out_file_path))
      for (dir <- dirs){
        val jpgs=subdirs2(new File(dir))
        jpgs.foreach(x => {val jpg_full_name=x.toPath().toString();writer.write(getImagePixel(jpg_full_name));writer.write("\n");println(jpg_full_name)})  
      }      
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
        //533,800=sep=-16745537,-16745794,-16746051,-16746565,-16747335,-16748105,-16........
        
        /*
         * test for split_rgb_line   测试解析getImagePixel输出的文本数据  用于hdfs读取
         * */       
        val demo2=split_rgb_line(demo)
        val height=demo2(0).length
        val width=demo2(0)(0).length
        println("width="+width)
        println("height="+height)
        println("rgb:")
        for(i <-0 to 10){
          for(j<-0 until demo2(0)(0).length){
            println("i="+i+",j="+j+",rgb=("+demo2(0)(i)(j)+","+demo2(1)(i)(j)+","+demo2(2)(i)(j)+")")
          }
        }
        dp_utils.gen_pict.gen_pict_fun(dp_utils.gen_pict.img_flatten_2d_to_1d(demo2(0)).map(x=>(x*255).toInt),
                                       dp_utils.gen_pict.img_flatten_2d_to_1d(demo2(1)).map(x=>(x*255).toInt),
                                       dp_utils.gen_pict.img_flatten_2d_to_1d(demo2(2)).map(x=>(x*255).toInt),
                                       width=width,height=height,jpg_out_path="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/test/001_split_rgb_line_out.jpg")
        
        /*
         * test for gen_hdfs_txt   用于在本地把jpg图片转化为rgb形式的数据字符串(每个图片一行数据),用于存入hdfs内  
         * 产生数据很大
         * */ 
        //gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590.txt")
        //gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//net_7876","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//net_7876.txt")         
        //gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//train//female","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590_female_train.txt") 
        //gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//train//male","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590_male_train.txt") 
        gen_hdfs_txts(Array("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//train_min//female","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//train_min//male"),"D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590_train_min.txt") 
        //gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//test//female","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590_female_test.txt") 
        //gen_hdfs_txt("D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//train//lfw_5590//test//male","D://youku_work//python//spark_python_scala//scala//workpace//deeplearning//dataset//dataset_face//hdfs//lfw_5590_male_test.txt")         

    }    
}