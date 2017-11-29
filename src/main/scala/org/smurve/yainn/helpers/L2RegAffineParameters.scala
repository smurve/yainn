package org.smurve.yainn.helpers

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.T
import org.nd4s.Implicits._

/**
  * L2 regularized autoupdating parameters for affine layers
  * Weights initialized accoring to He et al. 2015: https://arxiv.org/pdf/1502.01852.pdf
  * @param inputSize input size
  * @param outputSize output size
  * @param alpha relative weight of the regularizing L2 cost
  * @param seed random seed
  */
class L2RegAffineParameters(inputSize: Int, outputSize: Int, alpha: Double,
                            seed: Long, updater: Option[Updater]) extends SmartParameters {

  val W: T = Nd4j.randn(outputSize, inputSize, seed) * math.sqrt(2.0 / inputSize)
  val b: T = Nd4j.zeros(outputSize, 1)

  /**
    * update the weights by eta-fold of the gradients if 'updatable' is set to true
    * @param gradients the recent gradients from back prop
    */
  override def update(gradients: (T, T)): Unit =
      updater.foreach(u=>u.update(W, b,
        (gradients._1, gradients._2)))

  /**
    * Parameters may come at a cost, here the sum of the squared weights.
    * This is called L2 regularization. Note that the bias does not incur any cost
    * @return
    */
  override def cost: Double = (W * W).sumT * 0.5 * alpha

  /**
    * The derivative of the cost is obviously the weight matrix itself.
    * @return
    */
  override def dC_dw: T = W * alpha
}

