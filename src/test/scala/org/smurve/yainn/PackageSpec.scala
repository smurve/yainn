package org.smurve.yainn

import org.scalatest.{FlatSpec, ShouldMatchers}

class PackageSpec extends FlatSpec with ShouldMatchers {

  "equiv" should "compare labels and classifications correctly" in {
    equiv(t(.1,.4,.3), t(0,1,0)) shouldBe true
    equiv(t(.1,.3,.3), t(0,1,0)) shouldBe false
    equiv(t(0,0,0), t(0,1,0)) shouldBe false
    equiv(t(.4,.3,.3), t(0,1,0)) shouldBe false
  }
}
