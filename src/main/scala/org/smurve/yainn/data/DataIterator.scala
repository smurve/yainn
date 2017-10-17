package org.smurve.yainn.data

import org.smurve.yainn.T

trait DataIterator {

  def nextMiniBatch(): (T, T)

  def hasNext: Boolean

  def reset(): Unit

  def newTestData(numTests: Int): (T, T)
}
