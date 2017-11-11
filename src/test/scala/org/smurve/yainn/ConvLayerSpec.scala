package org.smurve.yainn

import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.components.{Affine, AutoUpdatingConv, Output}
import org.smurve.yainn.helpers.ConvParameters

class ConvLayerSpec extends FlatSpec with ShouldMatchers {

  "A conv layer" should "update correctly after fbp" in {

    val fields = t(
      1,1,1,1,
      2,2,2,2)

    val x = t(
      1,2,3,4,
      2,3,4,5,
      3,4,5,6,
      4,5,6,7,

      7,6,5,4,
      6,5,4,3,
      5,4,3,2,
      4,3,2,1
    ).T.reshape(2, 16).T

    val yb = t(430, 430, 430, 430).reshape(2,2)

    val b = t(0, 0)
    val W = Nd4j.ones(2, 18)

    val params = ConvParameters(2,2, 4, 4, fields, 1e-6, 0.0)

    val nn = AutoUpdatingConv("conv", params) !! Affine("hidden", W, b) !! Output(euc, euc_prime)

    nn.fp(x)(->, 0) shouldEqual t(432, 432)

    val back = nn.fbp(x, yb, x)

    back.C shouldEqual 4

    // the cost should decrease with each call to fbp, since the conv layer is auto-updating
    var c = 100.0
    for ( _ <- 0 to 20 ) {
      nn.fbp(x, yb, x).C should be < c
      c = nn.fbp(x, yb, x).C
    }

  }

}
