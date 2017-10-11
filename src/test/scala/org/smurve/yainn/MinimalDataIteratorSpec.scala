package org.smurve.yainn

import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.data.MinimalDataIterator

class MinimalDataIteratorSpec extends FlatSpec with ShouldMatchers {

  "An iterator " should "produce a well-defined number of mini-batches" in {

    val iterator = new MinimalDataIterator(0, 2, 3, 12345L)
    iterator.hasNext shouldBe true
    val b1 = iterator.nextMiniBatch()
    b1._1.shape shouldEqual Array(9,2)
    iterator.hasNext shouldBe true
    iterator.nextMiniBatch()
    iterator.hasNext shouldBe true
    iterator.nextMiniBatch()
    iterator.hasNext shouldBe false
    iterator.reset()
    val b4 = iterator.nextMiniBatch()
    b1 shouldEqual b4

  }
}
