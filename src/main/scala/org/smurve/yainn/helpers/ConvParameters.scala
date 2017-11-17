package org.smurve.yainn.helpers

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.T
import org.nd4s.Implicits._

/**
  * Parameter handling for convolutional layer - Rational:
  * The entire convolution can be seen as a multiplication with a large, sparse matrix.
  * W_eff and b_eff are the large-matrix representatives of the common weights and biases, respectively.
  *
  * @param h_Field the height of the receptive field
  * @param w_Field the width of the receptive field
  * @param h_Input the height of the input image
  * @param w_Input the width of the input image
  * @param n_Fields the number of fields
  */
class ConvParameters(val h_Field: Int, val w_Field: Int, val h_Input: Int, val w_Input: Int, val n_Fields: Int, alpha: Double, seed: Long,
                     updater: Option[Updater] = None) extends SmartParameters {

  private var fields = (Nd4j.rand(seed, n_Fields * h_Field * w_Field).T - 0.5) / 10.0
  private var biases = Nd4j.zeros(n_Fields).T


  val h_Output: Int = h_Input - h_Field + 1
  val w_Output: Int = w_Input - w_Field + 1
  // effective representation to support single-sparse-matrix-multiplication
  private val W_eff = Nd4j.zeros(n_Fields * h_Output * w_Output, h_Input * w_Input)
  private val b_eff = Nd4j.zeros(n_Fields * h_Output * w_Output,1)

  override def W: T = W_eff
  override def b: T = b_eff

  setParams((fields, Nd4j.zeros(n_Fields).T))


  /**
    * add deltas to the weights
    *
    * @param newWeights gradients for all weights
    */
  def setParams(newWeights: (T, T)): Unit = {
    fields = newWeights._1
    biases = newWeights._2
    for {
      n <- 0 until n_Fields
      ys <- 0 until h_Output
      xs <- 0 until w_Output
      yf <- 0 until h_Field
      xf <- 0 until w_Field
    } {
      val offset = n * h_Output * w_Output
      val r = offset + w_Output * ys + xs
      val c = w_Input * (ys + yf) + xf + xs
      W_eff(r, c) = newWeights._1(n * h_Field * w_Field + w_Field * yf + xf)
    }
    val n_output = h_Output * w_Output
    for {
      n <- 0 until n_Fields
    } {
      b_eff(n * n_output->(n+1) * n_output) = newWeights._2(n)
    }

  }

  override def update(gradients: (T, T)): Unit = {
    require(gradients._1.size(0) == fields.size(0))
    require(gradients._2.size(0) == biases.size(0))
    updater.foreach ( _.update(fields, biases, gradients))
    setParams((fields, biases))
  }

  /**
    * Parameters may come at a cost, here the sum of the squared weights.
    * This is called L2 regularization. Note that the bias does not incur any cost
    * @return the L2 based cost
    */
  override def cost: Double = (W * W).sumT * 0.5 * alpha

  /**
    * The derivative of the cost is obviously the weight matrix itself.
    * @return the cost derivative
    */
  override def dC_dw: T = W * alpha
}

object ConvParameters {

  /**
    * @param h_Field the height of the receptive field
    * @param w_Field the width of the receptive field
    * @param h_Input the height of the input image
    * @param w_Input the width of the input image
    * @param fields the column vector containing the receptive fields
    * @param seed a random seed
    * @return A ConvParameters instance
    */
  def apply(h_Field: Int, w_Field: Int, h_Input: Int, w_Input: Int, fields: T, eta: Double, alpha: Double, seed: Long): ConvParameters = {
    val n_fields = fields.size(0) / h_Field / w_Field
    val ret = new ConvParameters(h_Field, w_Field, h_Input, w_Input, n_fields, alpha, seed, Some(NaiveSGD(eta)))
    ret.setParams((fields, Nd4j.zeros(n_fields).T))
    ret
  }

}