package org.smurve.yainn.helpers

import org.smurve.yainn.T

trait SmartParameters {

  def W: T
  def b: T

  def update(gradients: (T, T)): Unit

  /**
    * Parameters may come at a cost, here the sum of the squared weights.
    * This is called L2 regularization. Note that the bias does not incur any cost
    * @return
    */
  def cost: Double

  /**
    * The derivative of the cost is obviously the weight matrix itself.
    * @return
    */
  def dC_dw: T
}
