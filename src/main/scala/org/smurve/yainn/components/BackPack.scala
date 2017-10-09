package org.smurve.yainn.components

import org.smurve.yainn.T

/**
  * A structure to hold and return the math artifacts for back propagation
  *
  * @param C: the current value of the cost function
  * @param dC_dy: the derivative of the cost function
  * @param grads: A list of all the gradients
  */
case class BackPack (C: Double, dC_dy: T, grads: List[(T,T)])

