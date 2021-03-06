package org.smurve

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.smurve.yainn.components.{Activation, Output}

package object yainn {

  /**
    * T, for 'Tensor', stands for any multi-dimensional algebraic structure.
    * INDArray is ND4J's high performance implementation of that algebraic concept
    */
  type T = INDArray

  /**
    * convenient tensor "literal" for column(!) vectors
    * @param arr varargs of components
    * @return
    */
  def t(arr: Double*): INDArray = Nd4j.create(Array(arr: _*)).T

  /**
    * convenient tensor "literal" for matrices consisting of vectors
    * @param arr varargs of components
    * @return
    */
  def tn(d: Int)(arr: Double*): INDArray = {
    val raw = Nd4j.create(Array(arr: _*))
    raw.reshape(raw.length / d, d).T
  }

  /**
    * The euclidean difference between two vectors
    * ...and its derivative
    */
  def euc(y: T, yb: T): Double = {
    val diff = y - yb
    (diff * diff).sumT[Double] * 0.5 / y.size(1)
  }
  def euc_prime(y: T, yb: T): T = (y - yb) / y.size(1)
  def Euclidean() = Output(euc, euc_prime)

  /**
    * cross-entropy cost, numerically unstable when y gets closer to yb.
    * Luckily, we don't need it for back prop other than monitoring the success
    */
  def x_ent(y: T, yb: T): Double = {
    val epsilon = 1e-10
    val yc = y * ( 1 - 2 * epsilon) + epsilon
    val res = (-yb * log(yc) + ( yb - 1) * log(-yc + 1)).sumT / y.size(1)
    res
  }

  /**
    * cross-entropy cost derivative, numerically stabilized
    */
  def x_ent_prime (y: T, yb: T): T = {
    val epsilon = 1e-30
    (yb - y) / (y * ( y - 1 ) + epsilon) / y.size(1)
  }

  /**
    * Softmax cross-entropy cost with logits
    */
  def smxewl(y: T, yb: T): Double = {
    (-yb * log(softmax(y.T).T)).sumT / yb.size(1)
  }
  def smxewl_prime (y: T, yb: T): T = (softmax(y.T).T - yb) / yb.size(1)
  def SoftMaxCrossEntropy(): Output = Output(smxewl, smxewl_prime)



  def sigmoid_prime(x: T): T = -sigmoid(x) * (sigmoid(x) - 1)
  def Sigmoid() = Activation(sigmoid, sigmoid_prime)

  def relu_prime(x: T): T = sign(relu(x))
  def Relu() = Activation(relu, relu_prime)



  /**
    * compare classifications with their respective true labels
    * @param classifications a matrix of vectors that shall be interpreted as classification
    * @param labels a matrix of one-hot vectors, with a 1 in a single position and 0s otherwise
    * @return true if the classification's max value is at the same index as the 1 in the label, false otherwise
    */
  def equiv(classifications: INDArray, labels: INDArray): INDArray = {
    val scale = t((0 until classifications.size(0)).map(_.toDouble).toArray: _*)
    val res = Nd4j.zeros(classifications.size(1))
    val arr = new Array[(T,T)](res.length)
    for ( i <- 0 until classifications.size(1)) {
      val classification = classifications(->, i -> (i + 1))
      val label = labels(->, i -> (i + 1))
      arr(i) = (classification, label)
      val max = classification.max(0).getDouble(0)
      val lbl_icx = (label.T ** scale).getDouble(0).toInt
      val nPreds = (classification === max).sum(0).getDouble(0)
      if (nPreds == 1 && max == classification.getDouble(lbl_icx)) res (i) = 1
    }
    res
  }


  def scaleToByte(min: Double, max: Double)(x: Double): Byte = {
    if (x < 0) 0 else {
      val min0 = math.max(0, min)
      (255 * (x - min0) / (max - min0)).toByte
    }
  }


  /**
    * visualize any 2-dim INDArray (e.g. an image) console-style.
    * @param x the matrix to be visualized
    * @return
    */
  def visualize(x: INDArray): String = {
    val hborder = " " + ("-" * 2 * x.size(0))
    require(x.rank == 2, "Can only visualize 2-dim arrays")
    val min: Double = x.minT[Double]
    val max: Double = x.maxT[Double]
    val img = (0 until x.size(0)).map(i => {
      val arr = toArray(x(i, ->))
      val row = arr.map(scaleToByte(min, max))
      rowAsString(row)
    }).mkString("\n")
    hborder + "\n" + img + "\n" + hborder
  }


  def rowAsString(bytes: Array[Byte]): String = {
    val res = bytes.map(b => {
      val n = b & 0xFF
      val c = if (n == 0) 0 else n / 32 + 1
      c match {
        case 0 => "  "
        case 1 => "' "
        case 2 => "''"
        case 3 => "::"
        case 4 => ";;"
        case 5 => "cc"
        case 6 => "OO"
        case 7 => "00"
        case 8 => "@@"
      }
    }).mkString("")
    "|" + res + "|"
  }

  /**
    * create a scala array from an INDArray
    * @param inda the ND4J Array to be converted
    */
  def toArray(inda: INDArray): Array[Double] = {
    val array = Array.fill(inda.length) (0.0)
    array.indices.foreach { index => array(index) = inda.getDouble(index)}
    array
  }

  /**
    * useful wrapper to measure execution time with little overhead
    * @param expression the expression to be executed
    * @tparam T the return type of the expression
    * @return a pair consisting of the actual result and the time spent to compute in millies
    */
  def timed[T](expression: => T): (T, Long) = {
    val in = System.currentTimeMillis()
    val res = expression
    val out = System.currentTimeMillis()
    (res, out - in)
  }

  /**
    * @return a stride-2 2x2 sum pooling matrix
    * @param dim the even dimension of a dim x dim input vector
    */
  def pool2by2(dim: Int): T = {
    val res = Nd4j.zeros(dim*dim/4, dim*dim)
    val d = dim/2
    for {
      r1 <- 0 until d
      r2 <- 0 until d
      y = d * r1 + r2
      x = 4 * r1 * d + 2 * r2
    } {
      res(y, x) = 1.0
      res(y, x+1) = 1.0
      res(y, x+2*d) = 1.0
      res(y, x+2*d+1) = 1.0
    }
    res
  }

  /**
    * @return a sharpened image by suppressing smaller pixel values
    * @param x  an image
    * @param cut the pixel value below which to filter
    */
  def sharpen(x: T, cut: Double): T = relu(x - cut)


  /**
    *
    */
  def shiftOp( n: Int, shiftBy: Int ): T = {
    val res = Nd4j.zeros(n, n)
    for { i <- 0 until n if i+shiftBy >= 0 && i+ shiftBy < n}{
      res(i, i+shiftBy) = 1.0
    }
    res
  }

  /**
    * supports parallalization of affine functions:
    * stack a row of ones on top of the matrix
    */
  def v1(x: T): T = Nd4j.hstack(Nd4j.ones(x.columns).T, x.T).T

  /**
    * supports parallalization of affine functions:
    * stack the given vector b left to the matrix W
    */
  def h(b: T, W: T): T = Nd4j.hstack(b, W)


  /**
    * experimental helpers
    */
  def ybt ( yb: T, m: Int): T = {
    val ym = t(0,1,2,3,4,5,6,7,8,9) === m
    val n = yb.size(1)
    val d = yb.size(0)
    val pro = ym.T ** yb
    val con = Nd4j.ones(n) - pro
    ym ** pro + (Nd4j.ones(d) - ym) ** con * 1.0 / (d - 1)
  }


  def euc_n (n: Int)( y: T, yb: T ) : Double = {
    euc ( y, ybt(yb, n))
  }

  def euc_n_prime (n: Int)( y: T, yb: T ) : T = {
    euc_prime( y, ybt(yb, n))
  }


}
