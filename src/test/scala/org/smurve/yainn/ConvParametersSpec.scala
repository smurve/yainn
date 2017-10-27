package org.smurve.yainn

import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.helpers.ConvParameters
import org.nd4s.Implicits._

class ConvParametersSpec extends FlatSpec with ShouldMatchers {

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


    val params = new ConvParameters(2, 2, 4, 4, fields, 1, 0.0)

    val res = params.W ** input

    res shouldEqual output
  }

}
