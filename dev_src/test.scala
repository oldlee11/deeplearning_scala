package dp_process_breeze
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
object test {
  def main(args: Array[String]) {
    val s=matrix_utils.build_2d_matrix((3,4))
    var ss=s+1.0
    for(i<-0 until ss.rows){
      for(j<-0 until ss.cols){
        print(ss(i,j)+"\t")
      }
      print("\n")
    }
    ss=(s+1.0)*3.0
    for(i<-0 until ss.rows){
      for(j<-0 until ss.cols){
        print(ss(i,j)+"\t")
      }
      print("\n")
    } 
    ss=ss.map(x=>x+1)
    for(i<-0 until ss.rows){
      for(j<-0 until ss.cols){
        print(ss(i,j)+"\t")
      }
      print("\n")
    }     
  }
}