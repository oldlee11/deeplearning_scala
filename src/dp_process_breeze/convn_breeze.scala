package dp_process_breeze


import breeze.linalg.DenseMatrix
/**
 * 
 * 2维图像的卷积和相关计算
 * 输出函数是        卷积  covon2d(x,kernel,convn_type)   
 *          相关  cor2d(x,kernel,cor_type)
 * 同 弄）use_code中的convn相比 做了优化 速度会更快
 *          
 * **/


object convn_breeze {
  
  
  /* 
   * 计算卷积
   * 注意卷积 于相关的差异是 卷积要先对kernel核做180翻转,相关则不用  
   * 
   * */  
  def convn2d(x1_in:DenseMatrix[Double],x2_in:DenseMatrix[Double],type_in:String):DenseMatrix[Double]={
    val rows_1:Int=x1_in.rows    //行数码=hight
    val cols_1:Int=x1_in.cols    //列数据=width
    val rows_2:Int=x2_in.rows    //行数码=hight
    val cols_2:Int=x2_in.cols    //列数据=width   
    
    if(type_in=="full"){
      val rows_out=rows_1+rows_2-1  //full  卷积后结果的行数
      val cols_out=cols_1+cols_2-1  //full  卷积后结果的列数
      var result:DenseMatrix[Double]=matrix_utils.build_2d_matrix((rows_out,cols_out))
      for(i<-0 until rows_2){
        for(j<-0 until cols_2){
          //向上扩展行
          var result_tmp:DenseMatrix[Double]=
              if(i!=0){
                DenseMatrix.vertcat(new DenseMatrix[Double](i,cols_1),x1_in*x2_in(i,j))
              }else{
                x1_in*x2_in(i,j)
              }
          //向下扩展行
          if(i!=(rows_2-1)){
            result_tmp=DenseMatrix.vertcat(result_tmp,new DenseMatrix[Double](rows_out-i-rows_1,cols_1))
          }
          //向左扩展列
          if(j!=0){
            result_tmp=DenseMatrix.horzcat(new DenseMatrix[Double](rows_out,j),result_tmp)
          }
          //向右扩展列
          if(j!=(cols_2-1)){
            result_tmp=DenseMatrix.horzcat(result_tmp,new DenseMatrix[Double](rows_out,cols_out-j-cols_1))
          }          
          result=result+ result_tmp
        }
      }
      result
    }else if((type_in=="valid") && (rows_1-rows_2+1>0) && (cols_1-cols_2+1>0)){
      convn2d(x1_in,x2_in,"full")(rows_2-1 until rows_1,cols_2-1 until cols_1)
    }else{
      null
    }
  }
  


  /* 
   * 计算相关
   * 注意卷积 于相关的差异是 卷积要先对kernel核做180翻转,相关则不用  
   * 
   * */  
  /* 
       版本1 
  */
  def cor2d(x1_in:DenseMatrix[Double],x2_in:DenseMatrix[Double],type_in:String):DenseMatrix[Double]={
    val rows_1:Int=x1_in.rows    //行数码=hight
    val cols_1:Int=x1_in.cols    //列数据=width
    val rows_2:Int=x2_in.rows    //行数码=hight
    val cols_2:Int=x2_in.cols    //列数据=width   
    
    if(type_in=="full"){
      val rows_out=rows_1+rows_2-1  //full  卷积后结果的行数
      val cols_out=cols_1+cols_2-1  //full  卷积后结果的列数
      var result:DenseMatrix[Double]=matrix_utils.build_2d_matrix((rows_out,cols_out))
      for(i<-0 until rows_2){
        for(j<-0 until cols_2){
          //向下扩展行
          var result_tmp:DenseMatrix[Double]=
              if(i!=0){
                //convn2d:DenseMatrix.vertcat(new DenseMatrix[Double](i,cols_1),x1_in*x2_in(i,j))
                DenseMatrix.vertcat(x1_in*x2_in(i,j),new DenseMatrix[Double](i,cols_1))
              }else{
                x1_in*x2_in(i,j)
              }
          //向上扩展行
          if(i!=(rows_2-1)){
            //convn2d:DenseMatrix.vertcat(result_tmp,new DenseMatrix[Double](rows_out-i-rows_1,cols_1))
            result_tmp=DenseMatrix.vertcat(new DenseMatrix[Double](rows_out-i-rows_1,cols_1),result_tmp)
          }
          //向右扩展列
          if(j!=0){
            //convn2d:DenseMatrix.horzcat(new DenseMatrix[Double](rows_out,j),result_tmp)
            result_tmp=DenseMatrix.horzcat(result_tmp,new DenseMatrix[Double](rows_out,j))
          }
          //向左扩展列
          if(j!=(cols_2-1)){
            //convn2d:DenseMatrix.horzcat(result_tmp,new DenseMatrix[Double](rows_out,cols_out-j-cols_1))
            result_tmp=DenseMatrix.horzcat(new DenseMatrix[Double](rows_out,cols_out-j-cols_1),result_tmp)
          }          
          result=result+ result_tmp
        }
      }
      result
    }else if((type_in=="valid") && (rows_1-rows_2+1>0) && (cols_1-cols_2+1>0)){
      cor2d(x1_in,x2_in,"full")(rows_2-1 until rows_1,cols_2-1 until cols_1)
    }else{
      null
    }
  }  
  /*
   * 版本2 使用rot90(x,2)但是不知道如何旋转
  def cor2d(x1_in:DenseMatrix[Double],x2_in:DenseMatrix[Double],type_in:String):DenseMatrix[Double]={
    convn2d(x1_in=x1_in,x2_in=rot90(x2_in,2),type_in=type_in)
  }
  */
  
  /*test*/
  def main(args: Array[String]) {
    /*
     * 2,1,1,1
     * 1,1,1,1
     * 1,1,1,2
     * */
    var x1:DenseMatrix[Double]=new DenseMatrix[Double](4,3,Array(2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0)).t
    print("x1=\n")
    for(i<-0 until x1.rows){
      for(j<-0 until x1.cols){
        print(x1(i,j)+"\t")
      }
      print("\n")
    }    
    
    
    
    /*
     * 1,2,3
     * 4,5,6
     * */                                      
    var x2:DenseMatrix[Double]=(new DenseMatrix[Double](3,2,Array(1.0,2.0,3.0,4.0,5.0,6.0))).t  
    print("x2=\n")
    for(i<-0 until x2.rows){
      for(j<-0 until x2.cols){
        print(x2(i,j)+"\t")
      }
      print("\n")
    }      
    
    
    print("test convn2d('full')\n")
    val test1=convn2d(x1,x2,"full")
    for(i<-0 until test1.rows){
      for(j<-0 until test1.cols){
        print(test1(i,j)+"\t")
      }
      print("\n")
    }
/*
test convn2d('full')
2.0	5.0	9.0	6.0	5.0	3.0	
9.0	17.0	27.0	21.0	16.0	9.0	
5.0	12.0	21.0	22.0	18.0	12.0	
4.0	9.0	15.0	19.0	16.0	12.0	
 * */
    
    
    print("test convn2d('valid')\n")
    val test2=convn2d(x1,x2,"valid")
    for(i<-0 until test2.rows){
      for(j<-0 until test2.cols){
        print(test2(i,j)+"\t")
      }
      print("\n")
    }  
/*
test convn2d('valid')
27.0	21.0	
21.0	22.0	
 * */ 
    
    
    print("test cor2d('full')\n")
    val test3=cor2d(x1,x2,"full")
    for(i<-0 until test3.rows){
      for(j<-0 until test3.cols){
        print(test3(i,j)+"\t")
      }
      print("\n")
    }
/*
12.0	16.0	19.0	15.0	9.0	4.0	
12.0	18.0	22.0	21.0	12.0	5.0	
9.0	16.0	21.0	27.0	17.0	9.0	
3.0	5.0	6.0	9.0	5.0	2.0	 
 * */    
    
    
    print("test cor2d('valid')\n")
    val test4=cor2d(x1,x2,"valid")
    for(i<-0 until test4.rows){
      for(j<-0 until test4.cols){
        print(test4(i,j)+"\t")
      }
      print("\n")
    }
/*
22.0	21.0	
21.0	27.0
 * */       
  }  
 }  