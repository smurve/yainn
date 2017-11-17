package org.smurve.yainn

import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.helpers.ConvParameters

class ConvParametersSpec extends FlatSpec with ShouldMatchers {

  val seed = 123

  val input: T = t(
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7)

  val output: T = t(
    8, 12, 16,
    12, 16, 20,
    16, 20, 24,
    16, 24, 32,
    24, 32, 40,
    32, 40, 48
  )

  val fields: T = t(1,1,1,1,2,2,2,2)

  "A ConvParameters" should "create correct convolution matrices" in {

    val params = ConvParameters(2, 2, 4, 4, fields, 1.0, 0.0, seed)

    val res = params.W ** input

    res shouldEqual output
  }


  it should "update parameters correctly" in {

    val params = ConvParameters(2, 2, 4, 4, fields, 1.0,0.0,  seed)
    val gradients = (t(2,2,2,2,3,3,3,3), t(1,2))
    params.update(gradients)
    params.b shouldEqual t(-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2,-2,-2,-2,-2,-2)
    val W = params.W
    W.sumT shouldEqual -72
  }

}
