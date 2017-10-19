package org.smurve.yainn.components

import org.nd4s.Implicits._
import org.smurve.yainn._

/**
  * A pre-processing layer, no gradients, no learning.
  */
case class ShrinkAndSharpen(name: String = "ShrinkAndSharpen", cut: Double = 0.2) extends AbstractLayer {

  private val Pool = pool2by2(28)

  def func(x: T): T = sharpen (Pool ** x * 0.25, cut )

  override def grads(x: T, dC_dy: T): Option[(T, T)] = None

  override def dC_dy(x: T, dC_dy_from_next: T): T = Pool.T ** dC_dy_from_next

  override def update(listOfDeltas: List[(T, T)]): Unit = {
    next.update(listOfDeltas)
  }
}

