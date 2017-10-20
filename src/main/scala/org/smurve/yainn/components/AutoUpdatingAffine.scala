package org.smurve.yainn.components

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn.T
import org.smurve.yainn.helpers.AdjustableAffineParameters

/**
  * A layer representing and affine function
  * @param p adjustable parameters
  */
case class AutoUpdatingAffine(name: String, p: AdjustableAffineParameters ) extends AbstractLayer {

  def func(x: T): T = {
    h(p.b, p.W) ** v1(x)
  }

  /**
    * back prop will not contain gradients of this layer.
    * @param x the input value
    * @param yb y_bar, the given true classification (label) for the input value x
    * @return a structure holding the backprop artifacts
    */
  override def fbp(x: T, yb: T): BackPack = {
    val from_next = next.fbp(func(x), yb)
    val dCdy = dC_dy(x, from_next.dC_dy)
    p.update(grads(x, from_next.dC_dy).get)

    BackPack(
      from_next.C,
      dCdy,
      from_next.grads)
  }

  override def grads(x: T, dC_dy: T): Option[(T, T)] = Some((dC_dy ** x.T, dC_dy.sum(1)))

  override def dC_dy(x: T, dC_dy_from_next: T): T = p.W.T ** dC_dy_from_next

  /** will not update but simply pass on to the next layer */
  override def update(listOfDeltas: List[(T, T)]): Unit = {
    next.update(listOfDeltas)
  }


  def v1(x: T): T = Nd4j.hstack(Nd4j.ones(x.columns).T, x.T).T
  def h(b: T, W: T): T = Nd4j.hstack(b, W)
}

