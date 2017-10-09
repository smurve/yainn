package org.smurve.yainn

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.scalactic.Equality
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.components.{Activation, Affine, Layer, Output}

/**
  * Gradient checking is an important quality gate in optimization attempts. Here we do the check with a tiny network with
  * two dense (or fully connected) layers for weights and biases.
  */
class GradientCheckSpec extends FlatSpec with ShouldMatchers {

  val seed = 1234L
  val N_x = 4
  val N_h = 3
  val N_y = 2
  val epsilon = 1e-3

  def sigmoid_prime(x: T): T = -sigmoid(x) * (sigmoid(x) - 1)

  def Sigmoid() = Activation(sigmoid, sigmoid_prime)

  def euc(x: T, x0: T): Double = ((x - x0).T ** (x - x0)).getDouble(0) * 0.5

  def euc_prime(x: T, x0: T): T = x - x0

  val W0: T = t(1, 2, 3, 4, .2, .3, .4, .5, .1, -.3, .4, -.5).reshape(N_h, N_x)
  val b0: T = t(.2, -1, .3)
  val W1: T = t(1, 2, 3, 4, .2, .3).reshape(N_y, N_h)
  val b1: T = t(-1, .3)

  val x1: T = t(-.1, .2, .4, .1) //, 1, 0, 1, 0, 1) // a cross
  val x2: T = t(-.2, .1, -.1, 0.3) //, 0, 1, 0, 1, 0) // a diamond
  val yb1: T = t(1, 0) // a label saying: cross
  val yb2: T = t(0, 1) // a label saying: diamond

  // the full network
  def nn(W0: T, b0: T, W1: T, b1: T): Layer = Affine(W0, b0) !! tail

  // the sub-network without the first affine layer
  val tail: Layer = Sigmoid() !! Affine(W1, b1) !! Sigmoid() !! Output(euc, euc_prime)


  /**
    * Numerical gradient computation suffers from the "difference of large numbers" problem, thus a deviation
    * of anything below 1% is just fine.
    */
  implicit val doubleEq: Equality[Double] = new Equality[Double] {
    override def areEqual(a: Double, b: Any): Boolean =
      math.abs(a - b.asInstanceOf[Double]) / (math.abs(a + b.asInstanceOf[Double]) + 1e-20) < 0.01
  }

  /**
    * matrix pairs plus/minus a little epsilon from the original at row r, col c
    */
  def Wlr(W: T, r: Int, c: Int, epsilon: Double): (T, T) = {
    val b = Nd4j.rand(W.shape, seed)
    val base = b === b(r, c) // this is a trick to get a matrix with value 1 at pos (r,c), 0 otherwise
    val res = (W - base * epsilon, W + base * epsilon)
    res
  }

  /**
    * vector pairs plus/minus a little epsilon from the original at row r, col c
    */
  def blr(W: T, i: Int, epsilon: Double): (T, T) = {
    Wlr(W, i, 0, epsilon)
  }


  "A minimal neural network" should "compute numerically plausible gradients in the first layer's weights" in {

    val grads0 = nn(W0, b0, W1, b1).fbp(x2, yb2).grads.head

    // We iterate through all indices of W0
    for (r <- 0 until N_h; c <- 0 until N_x) {

      // get 2 matrices with slightly tweaked parameter at that position
      val (wl, wr) = Wlr(W0, r, c, epsilon)

      // networks with slightly changed paramters
      val nn_l = Affine(wl, b0) !! tail
      val nn_r = Affine(wr, b0) !! tail

      // it's like comparing two networks and seeing which one's better!
      val gradW_num = (nn_r.fbp(x2, yb2).C - nn_l.fbp(x2, yb2).C) / 2 / epsilon

      gradW_num shouldEqual grads0._1.getDouble(r, c)
    }
  }

  "A minimal neural network" should "compute numerically plausible gradients in the first layer's bias" in {

    val grads0 = nn(W0, b0, W1, b1).fbp(x2, yb2).grads.head

    // We iterate through all indices of b0
    for (r <- 0 until N_h) {

      // get 2 matrices with slightly tweaked parameter at that position
      val (bl, br) = blr(b0, r, epsilon)

      // networks with slightly changed paramters
      val nn_l = Affine(W0, bl) !! tail
      val nn_r = Affine(W0, br) !! tail

      // it's like comparing two networks and seeing which one's better!
      val gradW_num = (nn_r.fbp(x2, yb2).C - nn_l.fbp(x2, yb2).C) / 2 / epsilon

      gradW_num shouldEqual grads0._2.getDouble(r)
    }
  }

  "A minimal neural network" should "compute numerically plausible gradients in the hidden layer's weights" in {

    val grads1 = nn(W0, b0, W1, b1).fbp(x2, yb2).grads(1)
    // We iterate through all indices of W1
    for (r <- 0 until N_y; c <- 0 until N_h) {

      // get 2 matrices with slightly tweaked parameter at that position
      val (wl, wr) = Wlr(W1, r, c, epsilon)

      // networks with slightly changed paramters
      val nn_l = Affine(W0, b0) !! Sigmoid() !! Affine(wl, b1) !! Sigmoid() !! Output(euc, euc_prime)
      val nn_r = Affine(W0, b0) !! Sigmoid() !! Affine(wr, b1) !! Sigmoid() !! Output(euc, euc_prime)

      // it's like comparing two networks and choosing the better one
      val gradW_num = (nn_r.fbp(x2, yb2).C - nn_l.fbp(x2, yb2).C) / 2 / epsilon

      gradW_num shouldEqual grads1._1.getDouble(r, c)
    }
  }

  "A minimal neural network" should "compute numerically plausible gradients in the hidden layer's bias" in {

    val grads1 = nn(W0, b0, W1, b1).fbp(x2, yb2).grads(1)
    // We iterate through all indices of W1
    for (r <- 0 until N_y) {

      // get 2 matrices with slightly tweaked parameter at that position
      val (bl, br) = blr(b1, r, epsilon)

      // networks with slightly changed paramters
      val nn_l = Affine(W0, b0) !! Sigmoid() !! Affine(W1, bl) !! Sigmoid() !! Output(euc, euc_prime)
      val nn_r = Affine(W0, b0) !! Sigmoid() !! Affine(W1, br) !! Sigmoid() !! Output(euc, euc_prime)

      // it's like comparing two networks and choosing the better one
      val gradW_num = (nn_r.fbp(x2, yb2).C - nn_l.fbp(x2, yb2).C) / 2 / epsilon

      gradW_num shouldEqual grads1._2.getDouble(r)
    }

  }

}
