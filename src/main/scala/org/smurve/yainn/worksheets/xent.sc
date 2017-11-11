import org.nd4j.linalg.ops.transforms.Transforms.log
import org.smurve.yainn._
import org.nd4s.Implicits._

def x_ent(y: T, yb: T): Double = {
  val epsilon = 1e-30
  val yc = y * ( 1 - 2 * epsilon) + epsilon
  (-yb.T ** log(yc) + ( yb - 1).T ** log(-yc + 1)).sumT
}

def x_ent_prime ( yy: T, yyb: T): T = {
  val epsilon = 1e-30
  val denom = yy * ( yy - 1 ) + epsilon
  (yyb - yy) / denom
}

val yb_ = t(1,0,0,0)
val y1_ = t(.4, .2, .2, .1)
val y2_ = t(.6, .1, .1, .1)
val y3_ = t(.7, .1, .0, .1)
x_ent(y1_, yb_)
x_ent(y2_, yb_)
x_ent(y3_, yb_)

val e0 = t(1e-4, 0, 0, 0)
val e1 = t(0, 1e-4, 0, 0)
val e2 = t(0, 0, 1e-4, 0)
val e3 = t(0, 0, 0, 1e-4)

val yr0 = y1_ + e0
val yl0 = y1_ - e0

val epsilon = 1e-30
val denom = y1_ * ( y1_ - 1 )
(yb_ - y1_) / denom

x_ent_prime(y1_, yb_)
val dy0 = (x_ent(y1_ + e0, yb_) - x_ent(y1_ - e0, yb_)) / 2 / 1e-4
val dy1 = (x_ent(y1_ + e1, yb_) - x_ent(y1_ - e1, yb_)) / 2 / 1e-4
val dy2 = (x_ent(y1_ + e2, yb_) - x_ent(y1_ - e2, yb_)) / 2 / 1e-4
val dy3 = (x_ent(y1_ + e3, yb_) - x_ent(y1_ - e3, yb_)) / 2 / 1e-4
