package org.smurve

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.smurve.yainn.components.Activation

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
    */
  def euc(y: T, yb: T): Double = {
    val diff = y - yb
    (diff * diff).sumT[Double] * 0.5
  }

  /**
    * ...and its derivative
    */
  def euc_prime(y: T, yb: T): T = y - yb

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


}
