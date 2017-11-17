package org.smurve.yainn

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.scalactic.Equality
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.yainn.components.{AutoUpdatingConv, _}
import org.smurve.yainn.helpers.ConvParameters

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

  /** Some test data */
  val W0: T = t(1, 2, 3, 4, .2, .3, .4, .5, .1, -.3, .4, -.5).reshape(N_h, N_x)
  val b0: T = t(.2, -1, .3)
  val W1: T = t(1, 2, 3, 4, .2, .3).reshape(N_y, N_h)
  val b1: T = t(-1, .3)

  val x1: T = t(-.1, .2, .4, .1) //, 1, 0, 1, 0, 1) // a cross
  val x2: T = t(-.2, .1, -.1, 0.3) //, 0, 1, 0, 1, 0) // a diamond
  val yb1: T = t(1, 0) // a label saying: cross
  val yb2: T = t(0, 1) // a label saying: diamond

  // the sub-network without the first affine layer
  val tail: Layer = Sigmoid() !! Affine("", W1, b1) !! Sigmoid() !! Output(euc, euc_prime)

  // the full network
  def nn(W0: T, b0: T, W1: T, b1: T): Layer = Affine("", W0, b0) !! tail


  "A minimal neural network" should "compute numerically plausible gradients in the first layer's weights" in {

    val grads0 = nn(W0, b0, W1, b1).fbp(x2, yb2, x2).grads.head

    // We iterate through all indices of W0
    for (r <- 0 until N_h; c <- 0 until N_x) {

      // get 2 matrices with slightly tweaked parameter at that position
      val (wl, wr) = Wlr(W0, r, c, epsilon)

      // networks with slightly changed paramters
      val nn_l = Affine("", wl, b0) !! tail
      val nn_r = Affine("", wr, b0) !! tail

      // it's like comparing two networks and seeing which one's better!
      val gradW_num = (nn_r.fbp(x2, yb2, x2).C - nn_l.fbp(x2, yb2, x2).C) / 2 / epsilon

      gradW_num shouldEqual grads0._1.getDouble(r, c)
    }
  }

  "A minimal neural network" should "compute numerically plausible gradients in the first layer's bias" in {

    val grads0 = nn(W0, b0, W1, b1).fbp(x2, yb2, x2).grads.head

    // We iterate through all indices of b0
    for (r <- 0 until N_h) {

      // get 2 matrices with slightly tweaked parameter at that position
      val (bl, br) = blr(b0, r, epsilon)

      // networks with slightly changed paramters
      val nn_l = Affine("", W0, bl) !! tail
      val nn_r = Affine("", W0, br) !! tail

      // it's like comparing two networks and seeing which one's better!
      val gradW_num = (nn_r.fbp(x2, yb2, x1).C - nn_l.fbp(x2, yb2, x2).C) / 2 / epsilon

      gradW_num shouldEqual grads0._2.getDouble(r)
    }
  }

  "A minimal neural network" should "compute numerically plausible gradients in the hidden layer's weights" in {

    val grads1 = nn(W0, b0, W1, b1).fbp(x2, yb2, x2).grads(1)
    // We iterate through all indices of W1
    for (r <- 0 until N_y; c <- 0 until N_h) {

      // get 2 matrices with slightly tweaked parameter at that position
      val (wl, wr) = Wlr(W1, r, c, epsilon)

      // networks with slightly changed paramters
      val nn_l = Affine("", W0, b0) !! Sigmoid() !! Affine("", wl, b1) !! Sigmoid() !! Output(euc, euc_prime)
      val nn_r = Affine("", W0, b0) !! Sigmoid() !! Affine("", wr, b1) !! Sigmoid() !! Output(euc, euc_prime)

      // it's like comparing two networks and choosing the better one
      val gradW_num = (nn_r.fbp(x2, yb2, x2).C - nn_l.fbp(x2, yb2, x2).C) / 2 / epsilon

      gradW_num shouldEqual grads1._1.getDouble(r, c)
    }
  }

  "A minimal neural network" should "compute numerically plausible gradients in the hidden layer's bias" in {

    val grads1 = nn(W0, b0, W1, b1).fbp(x2, yb2, x2).grads(1)
    // We iterate through all indices of W1
    for (r <- 0 until N_y) {

      // get 2 matrices with slightly tweaked parameter at that position
      val (bl, br) = blr(b1, r, epsilon)

      // networks with slightly changed paramters
      val nn_l = Affine("", W0, b0) !! Sigmoid() !! Affine("", W1, bl) !! Sigmoid() !! Output(euc, euc_prime)
      val nn_r = Affine("", W0, b0) !! Sigmoid() !! Affine("", W1, br) !! Sigmoid() !! Output(euc, euc_prime)

      // it's like comparing two networks and choosing the better one
      val gradW_num = (nn_r.fbp(x2, yb2, x2).C - nn_l.fbp(x2, yb2, x2).C) / 2 / epsilon

      gradW_num shouldEqual grads1._2.getDouble(r)
    }

  }

  "A convolutional neural network" should "compute numerically plausible gradients for multiple fields and images" in {

    val fields = t(
      1,2,
      2,3,
      3,4,
      2,3,
      3,4,
      4,5)
    val params = ConvParameters(3, 2, 4, 4, fields, 0.0, 0.0, seed)

    val nn = AutoUpdatingConv("conv", params) !! Output(euc, euc_prime)
    val x0 = t(
      1,2,3,4,
      2,3,4,5,
      3,4,5,6,
      4,5,6,7,
      4,5,6,7,
      3,4,5,6,
      2,3,4,5,
      1,2,3,4)

    val x = x0.T.reshape(2, 16).T

    // Output 2 feature maps of size 2x3 for each of 2 images = 24 values
    val yb = t(
      1,0,0,0,0,0,1,0,0,0,0,0,
      0,1,0,0,0,0,0,0,1,0,0,0).T.reshape(2,12).T

    val grads = nn.fbp(x, yb, x).grads.head

    for {
      r <- 0 until 12
    } {

      // get 2 matrices with slightly tweaked parameter at that position
      val (fl, fr) = blr(fields, r, epsilon)

      // networks with slightly changed paramters
      val p_l = ConvParameters(3, 2, 4, 4, fl, 0.0, 0.0, seed)
      val nn_l = AutoUpdatingConv("convl", p_l) !! Output(euc, euc_prime)
      val p_r = ConvParameters(3, 2, 4, 4, fr, 0.0, 0.0, seed)
      val nn_r = AutoUpdatingConv("convr", p_r) !! Output(euc, euc_prime)

      // it's like comparing two networks and choosing the better one
      val gradW_num = (nn_r.fbp(x, yb, x).C - nn_l.fbp(x, yb, x).C) / 2 / epsilon

      gradW_num shouldEqual grads._1.getDouble(r)
    }

  }

  /******************************************************************************
   *                  Helpers
   ******************************************************************************/

  /**
    * Numerical gradient computation suffers from the "difference of large numbers" problem, thus a deviation
    * of anything below 1% is just fine.
    */
  implicit lazy val doubleEq: Equality[Double] = new Equality[Double] {
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
    * column vector pairs plus/minus a little epsilon from the original at row r
    */
  def blr(b: T, r: Int, epsilon: Double): (T, T) = {
    Wlr(b, r, 0, epsilon)
  }



}
