package org.smurve.yainn.helpers

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.T
import org.nd4s.Implicits._

/**
  *
  * @param h_Field the height of the receptive field
  * @param w_Field the width of the receptive field
  * @param h_Input the height of the input image
  * @param w_Input the width of the input image
  * @param fields  the fields in a single vector, side by side
  */
case class ConvParameters(h_Field: Int, w_Field: Int, h_Input: Int, w_Input: Int, fields: T, eta: Double) extends SmartParameters {

  require(fields.size(0) % (h_Field * w_Field) == 0, "fields size not consistent")

  val n_Fields: Int = fields.size(0) / h_Field / w_Field
  val h_Output: Int = h_Input - h_Field + 1
  val w_Output: Int = w_Input - w_Field + 1

  private val W_eff = Nd4j.zeros(n_Fields * h_Output * w_Output, h_Input * w_Input)
  private val b_eff = Nd4j.zeros(n_Fields * h_Output * w_Output,1)

  override def W: T = W_eff
  override def b: T = b_eff

  setWeights(fields)


  /**
    * add deltas to the weights
    *
    * @param newWeights gradients for all weights
    */
  def setWeights(newWeights: T): Unit = {
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
      W_eff(r, c) = newWeights(n * h_Field * w_Field + w_Field * yf + xf)
    }
  }

  override def update(gradients: (T, T)): Unit = {
    require(gradients._1.size(0) == fields.size(0))
    require(gradients._2.size(0) == b_eff.size(0))
    fields.subi(gradients._1 * eta)
    setWeights(fields)
    b.subi(gradients._2 * eta)
  }

  /**
    * Parameters may come at a cost, here the sum of the squared weights.
    * This is called L2 regularization. Note that the bias does not incur any cost
    *
    * @return zero: No cost here
    */
  override def cost: Double = 0.0

  /**
    * The derivative of the cost is obviously the weight matrix itself.
    *
    * @return zeros, no cost here
    */
  override def dC_dw: T = {
    Nd4j.zeros(W.size(0), W.size(1))
  }
}

object ConvParameters {

  /**
    * get a new parameter set from random generator
    * @param h_Field the height of the receptive field
    * @param w_Field the width of the receptive field
    * @param h_Input the height of the input image
    * @param w_Input the width of the input image
    * @param n_Fields the number of different fields
    * @param seed a random seed
    * @return A ConvParameters instance
    */
  def apply(h_Field: Int, w_Field: Int, h_Input: Int, w_Input: Int, n_Fields: Int, eta: Double, seed: Long): ConvParameters = {
    val fields = (Nd4j.rand(seed, n_Fields * h_Field * w_Field) - 0.5) / 100
    new ConvParameters(h_Field, w_Field, h_Input, w_Input, fields.T, eta)
  }

}