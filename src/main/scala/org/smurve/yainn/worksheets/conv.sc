import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.{t => v}
import org.nd4s.Implicits._


val s = v(
  1,2,3,4,5,
  2,3,4,5,6,
  3,4,5,6,7,
  4,5,6,7,8,
  5,6,7,8,9)


val f = v(
  11, 12, 13,
  21, 22, 23,
  31, 32, 33)

val f1 = v(
  1, 1, 1,
  1, 1, 1,
  1, 1, 1)

val hs = 5
val ws = 5
val hf = 3
val wf = 3
val wt = ws - wf + 1
val ht = hs - hf + 1

val W = Nd4j.zeros(wt * ht, ws * hs)

for {
  ys <- 0 until hs - hf + 1
  xs <- 0 until ws - wf + 1
  yf <- 0 until hf
  xf <- 0 until wf
} {
  W(wt * ys + xs, ws * (ys + yf) + xf+ xs) =W(wt * ys + xs, ws * (ys + yf) + xf+ xs)  +  f1(wf *yf + xf)
}

W ** s

