import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._

type T = INDArray
val seed = 1234L
val epsilon = 1e-4
def t(arr: Double*): INDArray = Nd4j.create(Array(arr: _*)).T

val N_x = 4
val N_y = 2
val W0 = t(1,2,3,4,.2,.3,.4,.5).reshape(N_y, N_x)
val b0 = t(.2, -1)
val x1 = t(-.1, .2, .4, .1 )//, 1, 0, 1, 0, 1) // a cross
val x2 = t(-.2, .1, -.1, 0.3 ) //, 0, 1, 0, 1, 0) // a diamond
val yb1 = t(1, 0) // a label saying: cross
val yb2 = t(0, 1) // a label saying: diamond
val Phi: INDArray => INDArray = sigmoid
def Phi_prime (x: T) = - sigmoid(x) * (sigmoid(x) - 1)
def bb(n: Int) = {val res = Nd4j.zeros(N_y).T;res(n) = 1;res}
def bx(n: Int) = {val res = Nd4j.zeros(N_x).T;res(n) = 1;res}
def bW(r: Int, c: Int) = {val res = Nd4j.zeros(N_y, N_x);res(r,c) = 1; res}


def nabla_b (x: T, y_bar: T, W: T, b: T ) = {
  val z = W ** x + b
  val y = Phi(z)
  (y - y_bar) * Phi_prime(z)
}
def nabla_W (x: T, y_bar: T, W: T, b: T ) = {
  val z = W ** x + b
  val y = Phi(z)
  ((y - y_bar) * Phi_prime(z)) ** x.T
}
def D (x: T, y_bar: T, W: T, b: T ): Double = {
  val z = W ** x + b
  val diff = Phi(z) - y_bar
  (diff.T ** diff).getDouble(0) * 0.5
}

"Gradient checking Nabla b"
val (x0, yb0) = (x2, yb2)
val nb = nabla_b(x0, yb0, W0, b0)
val (nb0, nb1) = (nb.getDouble(0), nb.getDouble(1))
(D(x0, yb0, W0, b0 + bb(0) * epsilon) - D(x0, yb0, W0, b0 - bb(0) * epsilon)) / 2 / epsilon
(D(x0, yb0, W0, b0 + bb(1) * epsilon) - D(x0, yb0, W0, b0 - bb(1) * epsilon)) / 2 / epsilon

"Gradient checking nabla W"
nabla_W(x0, yb0, W0, b0)
nabla_W(x0, yb0, W0, b0).getDouble(0, 0)
(D(x0, yb0, W0 + bW(0,0) * epsilon, b0) - D(x0, yb0, W0 - bW(0,0) * epsilon, b0)) / 2 / epsilon

nabla_W(x0, yb0, W0, b0).getDouble(0, 1)
(D(x0, yb0, W0 + bW(0,1) * epsilon, b0) - D(x0, yb0, W0 - bW(0,1) * epsilon, b0)) / 2 / epsilon

nabla_W(x0, yb0, W0, b0).getDouble(0, 2)
(D(x0, yb0, W0 + bW(0,2) * epsilon, b0) - D(x0, yb0, W0 - bW(0,2) * epsilon, b0)) / 2 / epsilon

nabla_W(x0, yb0, W0, b0).getDouble(0, 3)
(D(x0, yb0, W0 + bW(0,3) * epsilon, b0) - D(x0, yb0, W0 - bW(0,3) * epsilon, b0)) / 2 / epsilon

nabla_W(x0, yb0, W0, b0).getDouble(1, 0)
(D(x0, yb0, W0 + bW(1,0) * epsilon, b0) - D(x0, yb0, W0 - bW(1,0) * epsilon, b0)) / 2 / epsilon

nabla_W(x0, yb0, W0, b0).getDouble(1, 1)
(D(x0, yb0, W0 + bW(1,1) * epsilon, b0) - D(x0, yb0, W0 - bW(1,1) * epsilon, b0)) / 2 / epsilon

nabla_W(x0, yb0, W0, b0).getDouble(1, 2)
(D(x0, yb0, W0 + bW(1,2) * epsilon, b0) - D(x0, yb0, W0 - bW(1,2) * epsilon, b0)) / 2 / epsilon

nabla_W(x0, yb0, W0, b0).getDouble(1, 3)
(D(x0, yb0, W0 + bW(1,3) * epsilon, b0) - D(x0, yb0, W0 - bW(1,3) * epsilon, b0)) / 2 / epsilon


val (i,j) = (1,3)
val orig = Nd4j.rand(10, 1).reshape(2,5)
val base = Nd4j.rand(orig.shape)
base === base(i,j)
