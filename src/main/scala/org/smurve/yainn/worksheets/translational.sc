import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.yainn._

type T = INDArray

val y = t(0.5, 0.0, 0.5, 0.1, 0.9, 0.3, 0.4, 0.0, 0.4, 0.1, 0.8, 0.2).T.reshape(4,3).T
val yperf = t(0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0).T.reshape(4,3).T

val yb = t(0,0,1,0,1,0, 1,0,0, 0, 1, 0).T.reshape(4,3).T

val yb2 = t(0,0,1)

val m = 0

val ym = t(0,1,2/*,3,4,5,6,7,8,9*/) === m
val n = yb.size(1)
val d = yb.size(0)
val pro = yb2.T ** yb
val con = Nd4j.ones(n) - pro
ym ** pro + (Nd4j.ones(d) - ym) ** con * 1.0 / (d - 1)


def ybt ( yb: T, m: Int): T = {
  val ym = t(0,1,2/*,3,4,5,6,7,8,9*/) === m
  val n = yb.size(1)
  val d = yb.size(0)
  val pro = ym.T ** yb
  val con = Nd4j.ones(n) - pro
  ym ** pro + (Nd4j.ones(d) - ym) ** con * 1.0 / (d - 1)
}

ybt(yb, 0)
ybt(yb, 1)
ybt(yb, 2)

def euc_m ( y:T, yb:T )(m: Int): Double = {
  val yb_ = ybt(yb, m)

  ((y - yb_) * (y-yb_)).sumT
}



def euc_1 ( y: T, yb: T) = euc_m(y, yb)(1)

val dist = euc_1(yperf, yb)

