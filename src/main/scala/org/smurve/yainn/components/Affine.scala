package org.smurve.yainn.components

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.T
import org.nd4s.Implicits._

/**
  * A layer representing and affine function
  * @param W the linear part (weight matrix)
  * @param b the translational part (bias)
  */
case class Affine(name: String, W: T, b: T) extends AbstractLayer {

  def func(x: T): T = {
    /* for a single input vector that would be expressed by
    W ** x + b
    // to allow parallelization of arbitrary batches, we apply a trick
    (b_1, w_11, w_12)   (  1   |  1   )   ( b_1 + w_11 * x_11 + w12 * x_21 | b_1 + w_11 * x_12 + w12 * x_22 )
    (b_2, w_21, w_22) * ( x_11 | x_12 ) = ( b_2 + w_21 * x_11 + w22 * x_21 | b_2 + w_21 * x_12 + w22 * x_22 )
    (b_3, w_31, w_32)   ( x_21 | x_22 )   ( b_3 + w_31 * x_11 + w32 * x_21 | b_3 + w_31 * x_12 + w32 * x_22 )
     */
    h(b, W) ** v1(x)
  }

  override def grads(x: T, dC_dy: T): Option[(T, T)] = Some((dC_dy ** x.T, dC_dy.sum(1)))

  override def dC_dy(x: T, dC_dy_from_next: T): T = W.T ** dC_dy_from_next

  // use Nd4j's inplace subtraction
  override def update(listOfDeltas: List[(T, T)]): Unit = {
    W.subi(listOfDeltas.head._1)
    b.subi(listOfDeltas.head._2)
    next.update(listOfDeltas.tail)
  }

  def v1(x: T): T = Nd4j.hstack(Nd4j.ones(x.columns).T, x.T).T
  def h(b: T, W: T): T = Nd4j.hstack(b, W)
}

