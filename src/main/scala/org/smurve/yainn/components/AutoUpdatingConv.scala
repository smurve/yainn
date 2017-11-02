package org.smurve.yainn.components

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn._
import org.smurve.yainn.helpers.ConvParameters

/**
  * A layer representing and affine function that updates automatically after each call to fbp()
  * Note that this layer considers a potential cost that may come from the weights.
  *
  * @param p adjustable parameters
  */
case class AutoUpdatingConv(name: String, p: ConvParameters) extends AbstractLayer {

  def func(x: T): T = {
    h(p.b, p.W) ** v1(x)
  }

  /**
    * back prop will not contain gradients of this layer.
    *
    * @param x  the input value
    * @param yb y_bar, the given true classification (label) for the input value x
    * @return a structure holding the backprop artifacts
    */
  override def fbp(x: T, yb: T, orig_x: T, update: Boolean = true): BackPack = {
    val from_next = next.fbp(func(x), yb, orig_x, update)
    val dCdy = dC_dy(x, from_next.dC_dy)

    val myGrads = grads(x, from_next.dC_dy).get
    if ( update ) {
      p.update(myGrads)
    }

    BackPack(
      from_next.C + p.cost,
      dCdy,
      List(myGrads) ::: from_next.grads)
  }

  override def grads(x: T, dC_dy: T): Option[(T, T)] = {

    val nabla_w = dC_dy ** x.T

    val nabla_f = Nd4j.zeros(p.n_Fields * p.h_Field * p.w_Field).T
    for {
      n <- 0 until p.n_Fields
      i <- 0 until p.h_Field
      j <- 0 until p.w_Field
    } {
      val d_df_ij = (for {
        r <- 0 until p.h_Output * p.w_Output
      } yield
        nabla_w(r + n*p.h_Output * p.w_Output, r % p.w_Output + r / p.w_Output * p.w_Input + i * p.w_Input + j)).sum

      nabla_f(n * p.h_Field * p.w_Field + i * p.w_Field + j) = d_df_ij
    }

    Some((nabla_f, dC_dy.sum(1)))
  }

  override def dC_dy(x: T, dC_dy_from_next: T): T = p.W.T ** dC_dy_from_next

  /** will not update but simply pass on to the next layer */
  override def update(listOfDeltas: List[(T, T)]): Unit = {
    next.update(listOfDeltas)
  }
}

