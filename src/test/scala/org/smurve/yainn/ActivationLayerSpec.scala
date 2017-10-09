package org.smurve.yainn

import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.components.{Activation, BackPack, Output}

class ActivationLayerSpec extends FlatSpec with ShouldMatchers {

  // the euclidean distance as a convenient cost function
  def cost(x:T, x0: T): Double = (( x - x0 ).T ** (x - x0)).getDouble(0) * 0.5
  def cost_prime(x: T, x0: T): T = x - x0
  val output = Output(cost, cost_prime)

  def fn(x:T): T = x * x // with the help of the Hadamard product
  def fnp(x:T): T = x * 2 // fn^prime

  // can't test without a terminal output layer
  private def activation() = Activation(fn, fnp) !! output

  "An activation layer" should "support forward pass by applying its function" in {
    activation().fp(t(2)) shouldEqual t(4)
  }

  it should "support chaining with the !! operator and next()" in {
    activation().next shouldEqual output
  }

  it should "support backpropagation by providing cost and dC_dy" in {
    activation().fbp(t(2), t(2)) shouldEqual BackPack(2, t(8), Nil)
  }

  it should "support dC_dy" in {
    activation().dC_dy(t(2), t(2)) shouldEqual t(8)
  }

  it should "provide empty grads()" in {
    activation().grads(t(2), t(2)) shouldBe empty
  }

}
