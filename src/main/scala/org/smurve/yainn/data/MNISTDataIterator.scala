package org.smurve.yainn.data
import org.nd4s.Implicits._
import org.smurve.yainn.T

class MNISTDataIterator(training: (T,T), miniBatchSize: Int, test: (T,T)) extends DataIterator{

  private var cursor = 0

  /**
    * provide the next mini-batch, i.e. a pair of INDArrays as (images, labels)
    */
  def nextMiniBatch (): (T, T) = {
    if (!hasNext) throw new IllegalStateException("Iterator exhausted. Call init() to start from scratch.")
    val batchImgs = training._1(->, cursor * miniBatchSize ->  (cursor+1) * miniBatchSize)
    val batchLbls = training._2(->, cursor * miniBatchSize ->  (cursor+1) * miniBatchSize)
    cursor = cursor + 1
    (batchImgs, batchLbls)
  }


  override def hasNext: Boolean = cursor < training._1.columns() / miniBatchSize

  override def reset(): Unit = cursor = 0

  override def newTestData(numTests: Int): (T, T) = (test._1(->, 0->numTests),test._2(->, 0->numTests))
}
