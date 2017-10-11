package org.smurve.yainn.data

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.{T, t}
import org.nd4s.Implicits._

/**
  * produces "noisy" versions of simple 9-bit symbols
  * @param noiseRatio the noise, typically < 0.3
  * @param miniBatchSize the size of the batches to be provided by nextBatch()
  * @param seed a random seed
  */
class MinimalDataIterator (noiseRatio: Double, miniBatchSize: Int, numMiniBatches: Int, seed: Long) extends Iterator {

  val imgs: T = Nd4j.zeros(9, miniBatchSize * numMiniBatches)
  val lbls: T = Nd4j.zeros(3, miniBatchSize * numMiniBatches)


  var cursor = 0

  val cross: T = t(
    1, 0, 1,
    0, 1, 0,
    1, 0, 1)
  val label_cross: T = t(1, 0, 0)

  val diamond: T = t(
    0, 1, 0,
    1, 0, 1,
    0, 1, 0)
  val label_diamond: T = t(0, 1, 0)

  val tee: T = t(
    1, 1, 1,
    0, 1, 0,
    0, 1, 0)
  val label_tee: T = t(0, 0, 1)

  val alphabet = List(
    (cross, label_cross),
    (diamond, label_diamond),
    (tee, label_tee)
  )

  val rng = new java.util.Random(seed)

  createData()
  /**
    * provide a list of pairs of symbols and labels
    */
  def createData(): Unit = {
    for (i <- 0 until numMiniBatches * miniBatchSize ) {
      val symbol = (rng.nextDouble() * alphabet.size).toInt
      val img = alphabet(symbol)._1
      val lbl = alphabet(symbol)._2
      imgs(->, i) = img.T
      lbls(->, i) = lbl.T
    }
    val noise = (Nd4j.rand(seed, imgs.length).reshape(imgs.rows,imgs.columns()) - 0.5) * noiseRatio
    imgs.addi(noise)
  }

  /**
    * provide a list of pairs of symbols and labels
    */
  def nextMiniBatch (): (T, T) = {
    if (!hasNext) throw new UnsupportedOperationException("Iterator exhausted. Call init() to start from scratch.")
    val batchImgs = imgs(--->, cursor * miniBatchSize ->  (cursor+1) * miniBatchSize)
    val batchLbls = lbls(--->, cursor * miniBatchSize ->  (cursor+1) * miniBatchSize)
    cursor = cursor + 1
    (batchImgs, batchLbls)
  }

  override def hasNext: Boolean = cursor < numMiniBatches

  override def reset(): Unit = cursor = 0

  override def newTestData ( numTests: Int): (T, T) = {
    val testImgs = Nd4j.zeros(9, numTests)
    val testLbls = Nd4j.zeros(3, numTests)
    for (i <- 0 until numTests ) {
      val symbol = (rng.nextDouble() * alphabet.size).toInt
      val img = alphabet(symbol)._1 + Nd4j.rand(9, 1, rng.nextLong())*noiseRatio
      val lbl = alphabet(symbol)._2
      testImgs(--->, i) = img.T
      testLbls(--->, i) = lbl.T
    }
    (testImgs, testLbls)
  }
}
