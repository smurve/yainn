package org.smurve.yainn.components

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits.{i, _}
import org.smurve.yainn.T
import org.smurve.yainn.helpers.{ConvParameters, SmartParameters}

/**
  * A layer representing and affine function that updates automatically after each call to fbp()
  * Note that this layer considers a potential cost that may come from the weights.
  * @param p adjustable parameters
  */
case class AutoUpdatingConv(name: String, p: ConvParameters ) extends AbstractLayer {

  def func(x: T): T = {
    h(p.b, p.W) ** v1(x)
  }

  /**
    * back prop will not contain gradients of this layer.
    * @param x the input value
    * @param yb y_bar, the given true classification (label) for the input value x
    * @return a structure holding the backprop artifacts
    */
  override def fbp(x: T, yb: T, orig_x: T): BackPack = {
    val from_next = next.fbp(func(x), yb, orig_x)
    val dCdy = dC_dy(x, from_next.dC_dy)
    p.update(grads(x, from_next.dC_dy).get )

    BackPack(
      from_next.C + p.cost,
      dCdy,
      from_next.grads)
  }

  override def grads(x: T, dC_dy: T): Option[(T, T)] = {

    val nabla_w = dC_dy ** x.T

    val nabla_f = Nd4j.zeros(p.n_Fields * p.h_Field * p.w_Field).T
    for {
      n <- 0 until p.n_Fields
      i <- 0 until p.h_Field
      j <- 0 until p.w_Field
    } {
      val d_dw_ij = (for {
        y <- 0 until p.h_Output
        x <- 0 until p.w_Output
      } yield nabla_w(i + j + x * p.w_Input, j + y * p.w_Output )).sum

      nabla_f(n * p.h_Field * p.w_Field + i * p.w_Field + j) = d_dw_ij
    }

    Some((nabla_f, dC_dy.sum(1)))
  }

  override def dC_dy(x: T, dC_dy_from_next: T): T = p.W.T ** dC_dy_from_next

  /** will not update but simply pass on to the next layer */
  override def update(listOfDeltas: List[(T, T)]): Unit = {
    next.update(listOfDeltas)
  }


  def v1(x: T): T = Nd4j.hstack(Nd4j.ones(x.columns).T, x.T).T
  def h(b: T, W: T): T = Nd4j.hstack(b, W)
}
