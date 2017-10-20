package org.smurve.yainn.helpers

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.T
import org.nd4s.Implicits._

class AdjustableAffineParameters(inputSize: Int, outputSize: Int, eta: Double, seed: Long) {

  val W: T = (Nd4j.rand(outputSize, inputSize, seed) - 0.5) / 10
  val b: T = (Nd4j.rand(outputSize, 1, seed) - 0.5) / 10

  def update(gradients: (T, T)): Unit = {

    W.subi(gradients._1 * eta)
    b.subi(gradients._2 * eta)
  }
}
