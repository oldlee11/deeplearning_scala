package dp_parallel_calculate


//2维图像的卷积和相关计算
object convn {
  /*****************
   * 
   * 串行示例,逻辑ok
   * 
   *****************/
  /*计算卷积*/  
  //注意卷积 于相关的差异是 卷积要先对kernel核做180翻转,相关则不用  
  def convn2d(x1_in:Array[Array[Double]],x2_in:Array[Array[Double]],type_in:String):Array[Array[Double]]={
    val rows_1:Int=x1_in.length    //行数码=hight
    val cols_1:Int=x1_in(0).length //列数据=width
    val rows_2:Int=x2_in.length    //行数码=hight
    val cols_2:Int=x2_in(0).length //列数据=width    
    if(type_in=="full"){
      val result:Array[Array[Double]]=Array.ofDim[Double](rows_1+rows_2-1,cols_1+cols_2-1)
      for(i<-0 until rows_1){
        for(j<-0 until cols_1){
          for(k<-0 until rows_2){
            for(m<-0 until cols_2){
              result(i+k)(j+m) +=x1_in(i)(j)*x2_in(k)(m)    
            }
          }
        }
      }
      result
    }else if(type_in=="valid"){
      if((rows_1-rows_2+1<0)||(cols_1-cols_2+1<0)){
        Array()
      }else{
        val result:Array[Array[Double]]=Array.ofDim[Double](rows_1-rows_2+1,cols_1-cols_2+1)
        for(i<-0 until rows_1-rows_2+1){
          for(j<-0 until cols_1-cols_2+1){
            for(k<-0 until rows_2){
              for(m<-0 until cols_2){
                result(i)(j) +=x1_in(i+k)(j+m)*x2_in(rows_2-1-k)(cols_2-1-m)    
              }
            }
          }
        }
        result
      }
    }else{
      Array()
    }
  }
  
  /*计算相关*/
  def cor2d(x1_in:Array[Array[Double]],x2_in:Array[Array[Double]],type_in:String):Array[Array[Double]]={
    val rows_1:Int=x1_in.length    //行数码=hight
    val cols_1:Int=x1_in(0).length //列数据=width
    val rows_2:Int=x2_in.length    //行数码=hight
    val cols_2:Int=x2_in(0).length //列数据=width    
    if(type_in=="full"){
      val result:Array[Array[Double]]=Array.ofDim[Double](rows_1+rows_2-1,cols_1+cols_2-1)
      for(i<-0 until rows_1){
        for(j<-0 until cols_1){
          for(k<-0 until rows_2){
            for(m<-0 until cols_2){
              result(i+k)(j+m) +=x1_in(i)(j)*x2_in(rows_2-1-k)(cols_2-1-m)    
            }
          }
        }
      }
      result
    }else if(type_in=="valid"){
      if((rows_1-rows_2+1<0)||(cols_1-cols_2+1<0)){
        Array()
      }else{
        val result:Array[Array[Double]]=Array.ofDim[Double](rows_1-rows_2+1,cols_1-cols_2+1)
        for(i<-0 until rows_1-rows_2+1){
          for(j<-0 until cols_1-cols_2+1){
            for(k<-0 until rows_2){
              for(m<-0 until cols_2){
                result(i)(j) +=x1_in(i+k)(j+m)*x2_in(k)(m)    
              }
            }
          }
        }
        result
      }
    }else{
      Array()
    }
  }  
  

  
  /*****************
   * 
   * 串行示例2,逻辑ok
   * 
   *****************/  
  //使用串行方式2来实现convn2d
  def convn2d_2(x1_in:Array[Array[Double]],x2_in:Array[Array[Double]],type_in:String):Array[Array[Double]]={
    val rows_1:Int=x1_in.length    //行数码=hight
    val cols_1:Int=x1_in(0).length //列数据=width
    val rows_2:Int=x2_in.length    //行数码=hight
    val cols_2:Int=x2_in(0).length //列数据=width   
    if(type_in=="full"){
      val rows_3:Int=rows_1+rows_2-1
      val cols_3:Int=cols_1+cols_2-1
      var result:Array[Array[Double]]=Array.ofDim[Double](rows_3,cols_3)
      for(i<-0 until result.length){
        for(j<-0 until result(0).length){
          for(k<-0 until rows_2){
            for(m<-0 until cols_2){ 
              if((i-k)>=0 && (i-k)<rows_1 && (j-m)>=0 && (j-m)<cols_1){
                result(i)(j) +=x1_in(i-k)(j-m)*x2_in(k)(m) 
              }
            }
          }
        }
      }
      result
    }else if(type_in=="valid"){
      if((rows_1-rows_2+1<0)||(cols_1-cols_2+1<0)){
        Array()
      }else{
        val result:Array[Array[Double]]=Array.ofDim[Double](rows_1-rows_2+1,cols_1-cols_2+1)
        for(i<-0 until rows_1-rows_2+1){
          for(j<-0 until cols_1-cols_2+1){
            for(k<-0 until rows_2){
              for(m<-0 until cols_2){
                result(i)(j) +=x1_in(i+k)(j+m)*x2_in(rows_2-1-k)(cols_2-1-m)    
              }
            }
          }
        }
        result
      }
    }else{
      Array()
    }
  }  
  
  //使用串行方式2来实现cor2d
  def cor2d_2(x1_in:Array[Array[Double]],x2_in:Array[Array[Double]],type_in:String):Array[Array[Double]]={
    val rows_1:Int=x1_in.length    //行数码=hight
    val cols_1:Int=x1_in(0).length //列数据=width
    val rows_2:Int=x2_in.length    //行数码=hight
    val cols_2:Int=x2_in(0).length //列数据=width    
    if(type_in=="full"){
      val rows_3:Int=rows_1+rows_2-1
      val cols_3:Int=cols_1+cols_2-1      
      var result:Array[Array[Double]]=Array.ofDim[Double](rows_3,cols_3)
      for(i<-0 until result.length){
        for(j<-0 until result(0).length){
          for(k<-0 until rows_2){
            for(m<-0 until cols_2){ 
              if((i-k)>=0 && (i-k)<rows_1 && (j-m)>=0 && (j-m)<cols_1){
                result(i)(j) +=x1_in(i-k)(j-m)*x2_in(rows_2-1-k)(cols_2-1-m)  
              }              
            }
          }
        }
      }
      result
    }else if(type_in=="valid"){
      if((rows_1-rows_2+1<0)||(cols_1-cols_2+1<0)){
        Array()
      }else{
        val result:Array[Array[Double]]=Array.ofDim[Double](rows_1-rows_2+1,cols_1-cols_2+1)
        for(i<-0 until rows_1-rows_2+1){
          for(j<-0 until cols_1-cols_2+1){
            for(k<-0 until rows_2){
              for(m<-0 until cols_2){
                result(i)(j) +=x1_in(i+k)(j+m)*x2_in(k)(m)                
              }
            }
          }
        }
        result
      }
    }else{
      Array()
    }
  }    
  
  /*****************
   * 
   * 串行实现
   * 
   *****************/   
  def convn2d_parallel(x1_in:Array[Array[Double]],x2_in:Array[Array[Double]],type_in:String):Array[Array[Double]]={
    val rows_1:Int=x1_in.length    //行数码=hight
    val cols_1:Int=x1_in(0).length //列数据=width
    val rows_2:Int=x2_in.length    //行数码=hight
    val cols_2:Int=x2_in(0).length //列数据=width   
    if(type_in=="full"){
      val rows_3:Int=rows_1+rows_2-1
      val cols_3:Int=cols_1+cols_2-1
      var result:Array[Array[Double]]=Array.ofDim[Double](rows_3,cols_3)
      for(i<-0 until result.length){
        for(j<-0 until result(0).length){
          /*
          for(k<-0 until rows_2){
            for(m<-0 until cols_2){ 
              if((i-k)>=0 && (i-k)<rows_1 && (j-m)>=0 && (j-m)<cols_1){
                result(i)(j) +=x1_in(i-k)(j-m)*x2_in(k)(m) 
              }
            }
          }
          */
          
        }
      }
      result
    }else if(type_in=="valid"){
      if((rows_1-rows_2+1<0)||(cols_1-cols_2+1<0)){
        Array()
      }else{
        val result:Array[Array[Double]]=Array.ofDim[Double](rows_1-rows_2+1,cols_1-cols_2+1)
        for(i<-0 until rows_1-rows_2+1){
          for(j<-0 until cols_1-cols_2+1){
            for(k<-0 until rows_2){
              for(m<-0 until cols_2){
                result(i)(j) +=x1_in(i+k)(j+m)*x2_in(rows_2-1-k)(cols_2-1-m)    
              }
            }
          }
        }
        result
      }
    }else{
      Array()
    }
  }
  
  /*test*/
  def main(args: Array[String]) {
    var x1:Array[Array[Double]]=Array(Array(2,1,1,1),
                                      Array(1,1,1,1),
                                      Array(1,1,1,2))
    var x2:Array[Array[Double]]=Array(Array(1,2,3),
                                      Array(4,5,6))   
    print("test convn2d('full')\n")
    convn2d(x1,x2,"full").foreach(x=>{x.foreach(x=>print(x+"\t"));print("\n")})
/*
2.0	5.0	9.0	6.0	5.0	3.0	
9.0	17.0	27.0	21.0	16.0	9.0	
5.0	12.0	21.0	22.0	18.0	12.0	
4.0	9.0	15.0	19.0	16.0	12.0	
*/
    
    print("test convn2d_2('full')\n")
    convn2d_2(x1,x2,"full").foreach(x=>{x.foreach(x=>print(x+"\t"));print("\n")})
/*
2.0	5.0	9.0	6.0	5.0	3.0	
9.0	17.0	27.0	21.0	16.0	9.0	
5.0	12.0	21.0	22.0	18.0	12.0	
4.0	9.0	15.0	19.0	16.0	12.0	
*/    
    
    print("test convn2d('valid')\n")
    convn2d(x1,x2,"valid").foreach(x=>{x.foreach(x=>print(x+"\t"));print("\n")}) 
/*
27.0	21.0	
21.0	22.0
 * */
    
    print("test cor2d('full')\n")
    cor2d(x1,x2,"full").foreach(x=>{x.foreach(x=>print(x+"\t"));print("\n")})
/*
12.0	16.0	19.0	15.0	9.0	4.0	
12.0	18.0	22.0	21.0	12.0	5.0	
9.0	16.0	21.0	27.0	17.0	9.0	
3.0	5.0	6.0	9.0	5.0	2.0	
 * */
    
    print("test cor2d_2('full')\n")
    cor2d_2(x1,x2,"full").foreach(x=>{x.foreach(x=>print(x+"\t"));print("\n")})
/*
12.0	16.0	19.0	15.0	9.0	4.0	
12.0	18.0	22.0	21.0	12.0	5.0	
9.0	16.0	21.0	27.0	17.0	9.0	
3.0	5.0	6.0	9.0	5.0	2.0	
 * */    
    
    print("test cor2d('valid')\n")
    cor2d(x1,x2,"valid").foreach(x=>{x.foreach(x=>print(x+"\t"));print("\n")})
/*
22.0	21.0	
21.0	27.0	
 * */
  }
}