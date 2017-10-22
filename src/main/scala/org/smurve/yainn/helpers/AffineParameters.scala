package org.smurve.yainn.helpers

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.T
import org.nd4s.Implicits._

class AffineParameters(inputSize: Int, outputSize: Int, eta: Double, alpha: Double, seed: Long, updatable: Boolean = true) {

  val W: T = (Nd4j.rand(outputSize, inputSize, seed) - 0.5) / 10
  val b: T = (Nd4j.rand(outputSize, 1, seed) - 0.5) / 10

  def update(gradients: (T, T)): Unit = {
    if ( updatable ) {
      W.subi(gradients._1 * eta)
      b.subi(gradients._2 * eta)
    }
  }

  /**
    * Parameters may come at a cost, here the sum of the squared weights.
    * This is called L2 regularization. Note that the bias does not incur any cost
    * @return
    */
  def cost: Double = (W * W).sumT * 0.5 * alpha

  /**
    * The derivative of the cost is obviously the weight matrix itself.
    * @return
    */
  def dC_dw: T = W * alpha
}
