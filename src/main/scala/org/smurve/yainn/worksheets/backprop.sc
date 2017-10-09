import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

type T = INDArray
def t(arr: Double*): INDArray = Nd4j.create(Array(arr: _*)).T

val seed = 1234L
val epsilon = 1e-4

val N_x = 4
val N_h = 3
val N_y = 2
val W0 = t(1, 2, 3, 4, .2, .3, .4, .5, .1, -.3, .4, -.5).reshape(N_h, N_x)
val b0 = t(.2, -1, .3)
val W1 = t(1, 2, 3, 4, .2, .3).reshape(N_y, N_h)
val b1 = t(-1, .3)

val x1 = t(-.1, .2, .4, .1) //, 1, 0, 1, 0, 1) // a cross
val x2 = t(-.2, .1, -.1, 0.3) //, 0, 1, 0, 1, 0) // a diamond
val yb1 = t(1, 0) // a label saying: cross
val yb2 = t(0, 1) // a label saying: diamond

/** the "Lego" bricks*/
def euc(y: T, yb: T): Double = {
  val diff = y - yb
  (diff.T ** diff).getDouble(0) * 0.5
}
def euc_prime(y: T, yb: T): T = y - yb

trait Layer {
  def func(x: T): T
  def fp(x: T): T
  def fbp(x: T, yb: T): (T, Double, List[(T, T)])
  def next: Layer
  def !!(rhs: Layer): Layer
  def dC_dy(x: T, dC_dy: T): T
  def grads(x: T, dC_dy: T): Option[(T, T)]
}

abstract class AbstractLayer extends Layer {
  var rhs: Option[Layer] = None

  def grads(x: T, dC_dy: T): Option[(T, T)] = None
  def fp(x: T): T = next.fp(func(x))
  def next = rhs.get
  def !!(other: Layer): Layer = {
    rhs = rhs.map(_ !! other).orElse(Some(other))
    this
  }
  def fbp(x: T, yb: T) = {
    val (dC_dy_from_next, cost, grads_from_next) = next.fbp(func(x), yb)
    ( dC_dy(x, dC_dy_from_next),
      cost,
      grads(x, dC_dy_from_next).toList ::: grads_from_next)
  }
}


/** Here are the concrete classes */
case class Affine(W: T, b: T) extends AbstractLayer {
  def func(x: T): T = W ** x + b
  override def grads(x: T, dC_dy: T) = Some((dC_dy ** x.T, dC_dy))
  override def dC_dy(x: T, dC_dy_from_next: T): T = W.T ** dC_dy_from_next
}

case class Activation(Phi: T => T, Phi_prime: T => T) extends AbstractLayer {
  def func(x: T) = Phi(x)
  override def dC_dy(x: T, dC_dy_from_next: T): T = dC_dy_from_next * Phi_prime(x)
}

case class Output(C: (T, T) => Double, derivC: (T, T) => T) extends Layer {
  def func(x: T) = x
  def fp(x: T) = x
  def fbp(y: T, yb: T) = (derivC(y, yb), C(y, yb), Nil)

  def !!(next: Layer) = notInOutput
  override def next = notInOutput
  override def dC_dy(x: T, dC_dy: T) = notInOutput
  override def grads(x: T, dC_dy: T) = notInOutput

  def notInOutput = throw new UnsupportedOperationException("Nothing shalt be next to the output layer.")
}

def Sigmoid() = Activation(sigmoid, x => -sigmoid(x) * (sigmoid(x) - 1))


/** gradient check */
val Ws_r = W0 + 0 // there's no simple copy() method...;-(Ws_r(1, 3) = Ws_r(1, 3) + epsilon
Ws_r(0, 0) = Ws_r(0, 0) + epsilon
val Ws_l = W0 + 0
Ws_l(0, 0) = Ws_l(0, 0) - epsilon

def tail() = Sigmoid() !! Affine(W1, b1) !! Sigmoid() !! Output(euc, euc_prime)
val nn1 = Affine(W0, b0) !! tail()
val nnr = Affine(Ws_r, b0) !! tail()
val nnl = Affine(Ws_l, b0) !! tail()

val grad_W0_c = (nnr.fbp(x2, yb2)._2 - nnl.fbp(x2, yb2)._2) / 2 / epsilon
val grad_W0_a = nn1.fbp(x2, yb2)._3.head._1.getDouble(0, 0)
