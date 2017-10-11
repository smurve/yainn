package org.smurve.yainn

import org.scalatest.{FlatSpec, ShouldMatchers}
import org.nd4s.Implicits._

class PackageSpec extends FlatSpec with ShouldMatchers {

  "equiv" should "compare labels and classifications correctly" in {
    equiv(t(0,0,0,.1,.4,.3).reshape(2,3).T, t(1, 0, 0, 0,1,0).reshape(2,3).T) shouldBe t(0, 1)

    equiv(t(.1,.4,.3), t(0,1,0)) shouldBe t(1)
    equiv(t(.1,.3,.3), t(0,1,0)) shouldBe t(0)
    equiv(t(0,0,0), t(0,1,0)) shouldBe t(0)
    equiv(t(.4,.3,.3), t(0,1,0)) shouldBe t(0)


  }
}
