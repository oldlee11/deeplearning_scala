package dp_process
import scala.util.Random
import scala.math

object utils {

def sigmoid(x: Double): Double = 1.0 / (1.0 + math.pow(math.E, -x))

def dsigmoid(x:Double): Double = x * (1.0 - x)

def tanh(x:Double):Double= math.tanh(x)

def dtanh(x:Double):Double= 1.0 - x * x

def ReLU(x:Double):Double=x * (if(x > 0.0) 1.0 else 0.0)

def dReLU(x:Double):Double=1.0 * (if(x > 0.0) 1.0 else 0.0)

def uniform(rng:Random,min: Double, max: Double): Double = {
   rng.nextDouble() * (max - min) + min
}

def binomial(rng:Random,n: Int, p: Double): Int = {
    if(p < 0 || p > 1) return 0
    var c: Int = 0
    var r: Double = 0.0
    var i: Int = 0
    for(i <- 0 until n) {
      r = rng.nextDouble()
      if(r < p) c += 1
    }
    return c
  }  
}