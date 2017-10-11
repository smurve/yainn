package org.smurve.yainn.data

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn._
import org.nd4s.Implicits._

class XorDataIterator(seed: Long) extends Iterator {

  val rng = new java.util.Random(seed)

  // these four points are not linearly separable
  val centers = Array(
    (t(0,0), t(1,0)),
    (t(2,2), t(1,0)),
    (t(0,2), t(0,1)),
    (t(2,0), t(0,1))
  )

  override def nextMiniBatch(): (T, T) = {

    val res = (0 until 100).map({_=>
      val center = centers((rng.nextDouble() * 4).toInt)
      (center._1 + (Nd4j.rand(2, 1, rng.nextLong())-.5)*1.5, center._2)
    }).toList

    ???
  }

  override def hasNext: Boolean = true

  override def reset(): Unit = ()

  override def newTestData(numTests: Int): (T, T) = nextMiniBatch()
}
