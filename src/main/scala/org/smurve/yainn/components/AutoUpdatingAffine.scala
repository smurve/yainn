package org.smurve.yainn.components

import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.helpers.SmartParameters

/**
  * A layer representing and affine function that updates automatically after each call to fbp()
  * Note that this layer considers a potential cost that may come from the weights.
  * @param p adjustable parameters
  */
case class AutoUpdatingAffine(name: String, p: SmartParameters ) extends AbstractLayer {

  def func(x: T): T = {
    h(p.b, p.W) ** v1(x)
  }

  /**
    * @param x the input value
    * @param yb y_bar, the given true classification (label) for the input value x
    * @return a structure holding the backprop artifacts
    */
  override def fbp(x: T, yb: T, orig_x: T, update: Boolean = true): BackPack = {
    val from_next = next.fbp(func(x), yb, orig_x, update)
    val dCdy = dC_dy(x, from_next.dC_dy)

    val myGrads = grads(x, from_next.dC_dy).get

    val cost = p.cost

    if ( update )
      p.update(myGrads)

    BackPack(
      from_next.C + cost,
      dCdy,
      List(myGrads) ::: from_next.grads)
  }

  override def grads(x: T, dC_dy: T): Option[(T, T)] = Some((dC_dy ** x.T + p.dC_dw, dC_dy.sum(1)))

  override def dC_dy(x: T, dC_dy_from_next: T): T = p.W.T ** dC_dy_from_next

  /** will not update but simply pass on to the next layer */
  override def update(listOfDeltas: List[(T, T)]): Unit = {
    next.update(listOfDeltas)
  }

}

