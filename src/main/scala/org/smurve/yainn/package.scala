package org.smurve

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.sigmoid
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
    * The euclidean difference between two vectors
    */
  def euc(y: T, yb: T): Double = {
    val diff = y - yb
    (diff.T ** diff).getDouble(0) * 0.5
  }

  /**
    * ...and its derivative
    */
  def euc_prime(y: T, yb: T): T = y - yb

  def sigmoid_prime(x: T): T = -sigmoid(x) * (sigmoid(x) - 1)
  def Sigmoid() = Activation(sigmoid, sigmoid_prime)

  /**
    * compare a classification with a label
    * @param classification any vector that could be interpreted as classification
    * @param label a one-hot vector, with a 1 in a single position and 0s otherwise
    * @return true if the classification's max value is at the same index as the 1 in the label, false otherwise
    */
  def equiv(classification: INDArray, label: INDArray): Boolean = {
    val max = classification.max(0).getDouble(0)
    val scale = t(( 0 until label.size(0)).map(_.toDouble).toArray: _*)
    val lbl_icx = (label.T ** scale).getDouble(0).toInt
    val nPreds = (classification === max).sum(0).getDouble(0)
    val res = nPreds == 1 && max == classification.getDouble(lbl_icx)
    res
  }


}
