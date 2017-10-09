package org.smurve.yainn.components

import org.smurve.yainn.T

/**
  * An output layer
 *
  * @param C the cost function
  * @param derivC the derivative of the cost function with respect to its immediate input
  */
case class Output(C: (T, T) => Double, derivC: (T, T) => T) extends Layer {

  def func(x: T): T = x
  def fp(x: T): T = x
  def fbp(y: T, yb: T): BackPack = BackPack(C(y, yb), derivC(y, yb), Nil)

  def !!(next: Layer): Layer = unsupported
  override def next: Layer = unsupported
  override def grads(x: T, dC_dy: T): Option[(T,T)] = unsupported
  override def dC_dy(x: T, dC_dy: T): T = unsupported
  override def update(grads: List[(T, T)]): Unit = {}

  private def unsupported =
    throw new UnsupportedOperationException("Thou shalt not call this method for the output layer!")

}

