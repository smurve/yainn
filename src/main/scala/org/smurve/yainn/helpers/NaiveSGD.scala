package org.smurve.yainn.helpers

import org.nd4s.Implicits._
import org.smurve.yainn.T

/**
  * simple and naive SGD updating with constant learning rate eta
  * @param eta learning rate
  */
case class NaiveSGD ( eta: Double ) extends Updater {

  override def update( W: T, b: T, gradients: (T,T)): Unit = {
    W.subi(gradients._1 * eta)
    b.subi(gradients._2 * eta)
  }
}
