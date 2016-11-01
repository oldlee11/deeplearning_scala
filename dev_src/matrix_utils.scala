package dp_process_breeze

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

object matrix_utils {
  /*
   * 建立一个4维的矩阵,初始值=0.0
   * 内部数值是Double
   * 返回后是 DenseMatrix[DenseMatrix[Double]]
   */
  def build_4d_matrix(size_in:(Int,Int,Int,Int)):DenseMatrix[DenseMatrix[Double]]={
    DenseMatrix.fill(rows=size_in._1,cols=size_in._2)(v=DenseMatrix.zeros[Double](size_in._3,size_in._4))   
    /*
    val result:DenseMatrix[DenseMatrix[Double]]=new DenseMatrix(rows=size_in._1, cols=size_in._2, data=new Array[DenseMatrix[Double]](size_in._1*size_in._2))
    for(i <-0 until size_in._1){
      for(j <-0 until size_in._2){
        //DenseMatrix.create(rows=size_in._3,cols=size_in._4,data=new Array(size_in._3*size_in._4))
        result(i,j)=DenseMatrix.zeros[Double](size_in._3,size_in._4)
      }
    }
    result
    */
  }
  
  /*
   * 建立一个3维的矩阵,初始值=0.0
   * 内部数值是Double
   * 返回后是 DenseVector[DenseMatrix[Double]]
   */
  def build_3d_matrix(size_in:(Int,Int,Int)):DenseVector[DenseMatrix[Double]]={
    DenseVector.fill(size=size_in._1)(v=DenseMatrix.zeros[Double](size_in._2,size_in._3))
  }  

  /*
   * 建立一个2维的矩阵,初始值=0.0
   * 内部数值是Double
   * 返回后是 DenseMatrix[Double]
   */  
  def build_2d_matrix(size_in:(Int,Int)):DenseMatrix[Double]={
    DenseMatrix.zeros[Double](size_in._1,size_in._2)
  } 

  /*
   * 建立一个1维的矩阵,初始值=0.0
   * 内部数值是Double
   * 返回后是 DenseVector[Double]
   */  
  def build_1d_matrix(size_in:Int):DenseVector[Double]={
    DenseVector.zeros[Double](size_in)
  }   
  
  
  /*
   * trans_array2breeze_4d
   * 把4维的array转化为breeze的matrix
   * */
  def trans_array2breeze_4d(array_4d_in:Array[Array[Array[Array[Double]]]]):DenseMatrix[DenseMatrix[Double]]={
    val (len_1d,len_2d,len_3d,len_4d)=(array_4d_in.length,array_4d_in(0).length,array_4d_in(0)(0).length,array_4d_in(0)(0)(0).length)
    val result=build_4d_matrix((len_1d,len_2d,len_3d,len_4d))
    for(i<-0 until len_1d){
      for(j<-0 until len_2d){  
        for(m<-0 until len_3d){
          for(n<-0 until len_4d){
            result(i,j)(m,n)=array_4d_in(i)(j)(m)(n)
          }
        }
      }
    }
    result
  }
  
  
  /*
   * trans_array2breeze_3d
   * 把3维的array转化为breeze的matrix
   * */
  def trans_array2breeze_3d(array_3d_in:Array[Array[Array[Double]]]):DenseVector[DenseMatrix[Double]]={
    val (len_1d,len_2d,len_3d)=(array_3d_in.length,array_3d_in(0).length,array_3d_in(0)(0).length)
    val result=build_3d_matrix((len_1d,len_2d,len_3d))
    for(i<-0 until len_1d){
      for(j<-0 until len_2d){  
        for(m<-0 until len_3d){
          result(i)(j,m)=array_3d_in(i)(j)(m)
        }
      }
    }
    result
  }
  

  /*
   * trans_array2breeze_2d
   * 把2维的array转化为breeze的matrix
   * */
  def trans_array2breeze_2d(array_2d_in:Array[Array[Double]]):DenseMatrix[Double]={
    val (len_1d,len_2d)=(array_2d_in.length,array_2d_in(0).length)
    val result=build_2d_matrix((len_1d,len_2d))
    for(i<-0 until len_1d){
      for(j<-0 until len_2d){  
        result(i,j)=array_2d_in(i)(j)
      }
    }
    result
  }  
  
  //test
  def main(args: Array[String]) {
    //test for build_4d_matrix
    val demo=build_4d_matrix((1,2,3,4))
    for(i<-0 until 3){
      for(j<-0 until 4){
        print(demo(0,1)(i,j)+"\t")
      }
      print("\n")
    }
    //test for build_3d_matrix
    print(build_3d_matrix((2,3,4))(0)(0,0)+"\n")
    
  }
}
