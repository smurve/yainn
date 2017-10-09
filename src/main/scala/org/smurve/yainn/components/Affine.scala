package org.smurve.yainn.components

import org.smurve.yainn.T
import org.nd4s.Implicits._

/**
  * A layer representing and affine function
  * @param W the linear part (weight matrix)
  * @param b the translational part (bias)
  */
case class Affine(W: T, b: T) extends AbstractLayer {

  def func(x: T): T = W ** x + b

  override def grads(x: T, dC_dy: T) = Some((dC_dy ** x.T, dC_dy))

  override def dC_dy(x: T, dC_dy_from_next: T): T = W.T ** dC_dy_from_next

  // use Nd4j's inplace subtraction
  override def update(listOfDeltas: List[(T, T)]): Unit = {
    W.subi(listOfDeltas.head._1)
    b.subi(listOfDeltas.head._2)
    next.update(listOfDeltas.tail)
  }
}

