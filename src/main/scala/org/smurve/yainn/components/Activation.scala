package org.smurve.yainn.components

import org.smurve.yainn.T
import org.nd4s.Implicits._

/**
  * An activation function modelled as a layer in its own right.
  * @param Phi the component-wise function
  * @param Phi_prime the derivative of this function
  */
case class Activation(Phi: T => T, Phi_prime: T => T) extends AbstractLayer {

  def func(x: T) = Phi(x)

  /**
    * cost derivative with respect to this layer's input. Note that this method essentially implements the chain rule
    * @param x this layer's input
    * @param dC_dy_from_next cost derivative with respect to the next layer's input
    * @return The cost's total derivative up until this layer
    */
  override def dC_dy(x: T, dC_dy_from_next: T): T = dC_dy_from_next * Phi_prime(x)
}

