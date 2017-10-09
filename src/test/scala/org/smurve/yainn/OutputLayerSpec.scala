package org.smurve.yainn

import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.components.{BackPack, Output}

class OutputLayerSpec extends FlatSpec with ShouldMatchers {

  // when it doesn't matter
  val any: T = t(0)

  // the euclidean distance as a convenient cost function
  def cost(x:T, x0: T): Double = (( x - x0 ).T ** (x - x0)).getDouble(0) * 0.5
  def cost_prime(x: T, x0: T): T = x - x0

  val output = Output(cost, cost_prime)

  "An output layer" should "support a cost function" in {
    output.C(t(3), t(2)) shouldEqual 0.5
  }

  it should "support a cost derivative" in {
    output.derivC(t(3),t(2)) shouldEqual t(1)
  }

  it should "forward pass by simply returning the input" in {
    output.fp(t(1)) shouldEqual t(1)
  }

  it should "support backpropagation by providing cost and dC_dy" in {
    output.fbp(t(3), t(2)) shouldEqual BackPack(0.5, t(1), Nil)
  }

  it should "not support the !! operator" in {
    a[UnsupportedOperationException] mustBe thrownBy { output.!!(output) }
  }
  it should "not support next()" in {
    a[UnsupportedOperationException] mustBe thrownBy { output.next }
  }
  it should "not support dC_dy" in {
    a[UnsupportedOperationException] mustBe thrownBy { output.dC_dy(any, any)}
  }
  it should "not support grads()" in {
    a[UnsupportedOperationException] mustBe thrownBy { output.grads(any, any) }
  }


}
