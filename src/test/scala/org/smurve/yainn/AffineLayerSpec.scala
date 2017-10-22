package org.smurve.yainn

import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.components.{Activation, Affine, BackPack, Output}

class AffineLayerSpec extends FlatSpec with ShouldMatchers {

  // the euclidean distance as a convenient cost function
  def cost(x:T, x0: T): Double = (( x - x0 ).T ** (x - x0)).getDouble(0) * 0.5
  def cost_prime(x: T, x0: T): T = x - x0
  val output = Output(cost, cost_prime)

  val W: T = t(1,2,-2,-1).reshape(2,2)
  val b: T = t(1,-1)

  // can't test without a terminal output layer
  private def dense() = Affine("", W, b) !! output

  "An activation layer" should "support forward pass by applying its function" in {
    dense().fp(t(2, 3)) shouldEqual t(9,-8)
  }

  "An activation layer" should "support forward pass for arbitrary batches" in {
    dense().fp(tn(2)(2, 3, 2, 3)) shouldEqual tn(2)(9,-8, 9, -8)
  }

  it should "support chaining with the !! operator and next()" in {
    dense().next shouldEqual output
  }

  it should "support backpropagation by providing cost and dC_dy" in {
    val cost = 2.5
    val dC_dy = t(-5,-4)
    val gradW = t(-2, -3, 4, 6).reshape(2,2)
    val gradb = t(-1, 2)
    dense().fbp(t(2,3), t(10,-10), t(2,3)) shouldEqual BackPack(cost, dC_dy, List((gradW, gradb)))
  }

  it should "support dC_dy" in {
    dense().dC_dy(t(2,3), t(-1, 2)) shouldEqual t(-5, -4)
  }

  it should "support dC_dy for arbitrary batches" in {
    dense().dC_dy(tn(2)(2,3,2,3), tn(2)(-1, 2,-1, 2)) shouldEqual tn(2)(-5, -4,-5, -4)
  }

  it should "provide correct grads()" in {
    val gradW = t(-2, -3, 4, 6).reshape(2,2)
    val gradb = t(-1, 2)
    dense().grads(t(2, 3), t(-1, 2)) shouldBe Some(gradW, gradb)
  }

}
