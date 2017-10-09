package org.smurve.yainn.data

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.{T, t}
import org.nd4s.Implicits._

/**
  * produces "noisy" versions of simple 9-bit symbols
  * @param noise the noise, typically < 0.3
  * @param defaultBatchSize the size of the batches to be provided by nextBatch()
  * @param seed a random seed
  */
class MinimalDataIterator (noise: Double, defaultBatchSize: Int, seed: Long) {

  val cross: T = t(
    1, 0, 1,
    0, 1, 0,
    1, 0, 1)
  val label_cross: T = t(1, 0)

  val diamond: T = t(
    0, 1, 0,
    1, 0, 1,
    0, 1, 0)
  val label_diamond: T = t(0, 1)

  val tee: T = t(
    1, 1, 1,
    0, 1, 0,
    0, 1, 0)
  val label_tee: T = t(0, 1)

  val alphabet = List(
    (cross, label_cross),
    (diamond, label_diamond),
    (tee, label_tee)
  )

  val rng = new java.util.Random(seed)

  /**
    * provide a list of pairs of symbols and labels
    * @param batchSize override the default batch size if you wish
    */
  def nextBatch( batchSize: Int = defaultBatchSize): List[(T, T)] = {
    (1 to batchSize).map (_ => {
      val symbol = (rng.nextDouble() * alphabet.size).toInt
      (alphabet(symbol)._1 + Nd4j.rand(9, 1, rng.nextLong()), alphabet(symbol)._2)
    }).toList
  }
}
