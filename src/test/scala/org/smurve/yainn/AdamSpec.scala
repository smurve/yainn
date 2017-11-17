package org.smurve.yainn

import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.helpers.Adam
import org.nd4s.Implicits._

class AdamSpec extends FlatSpec with ShouldMatchers {


  "An Adam Optimizer" should "" in {

    val W = t(10, 10, 10, 10)
    val b = t(20, 20)

    val adam = Adam(size_W = 4, size_b = 2, eta = 1, beta2 = 0.9)

    adam.update(W, b, (t(-1,-1,-1,-1), t(-1, -1)))
    adam.update(W, b, (t(-1,-1,-1,-1), t(1, 1)))
    adam.update(W, b, (t(-1,-1,-1,-1), t(-1, -1)))
    adam.update(W, b, (t(-1,-1,-1,-1), t(1, 1)))
    adam.update(W, b, (t(-1,-1,-1,-1), t(-1, -1)))
    adam.update(W, b, (t(-1,-1,-1,-1), t(1, 1)))

    W shouldEqual t(4,4,4,4)
    b(0) should be > 18.0

    adam.update(W, b, (t(1,1,1,1), t(1, 1)))
    adam.update(W, b, (t(-1,-1,-1,-1), t(1, 1)))
    adam.update(W, b, (t(1,1,1,1), t(1, 1)))
    adam.update(W, b, (t(-1,-1,-1,-1), t(1, 1)))
    adam.update(W, b, (t(1,1,1,1), t(1, 1)))
    adam.update(W, b, (t(-1,-1,-1,-1), t(1, 1)))
    adam.update(W, b, (t(1,1,1,1), t(1, 1)))
    adam.update(W, b, (t(-1,-1,-1,-1), t(1, 1)))

    W(0) should be > 3.0
    b(0) should be < 14.0
  }
}
