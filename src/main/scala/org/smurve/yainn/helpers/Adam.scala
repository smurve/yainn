package org.smurve.yainn.helpers

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn.T
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._

/**
  * Adaptive Moment Updater, see Kingma, Ba 2014: https://arxiv.org/abs/1412.6980
  * @param beta1 Adams beta_1 momentum parameter, defaults to
  * @param beta2 Adams beta_2 other parameter, defaults to
  * @param eta learning rate
  * @param epsilon stability
  */
case class Adam ( beta1: Double = 0.9, beta2: Double = 0.999, eta: Double, epsilon: Double = 1e-8,
                  size_W: Int, size_b: Int ) extends Updater {

  private val v_W: T = Nd4j.zeros(size_W).T
  private val s_W: T = Nd4j.zeros(size_W).T
  private val v_b: T = Nd4j.zeros(size_b).T
  private val s_b: T = Nd4j.zeros(size_b).T
  private var t = 1

  /**
    * update the parameters
    * @param W weight parameters
    * @param b bias parameters
    * @param gradients gradients lined up as a pair of two column vectors
    */
  override def update( W: T, b: T, gradients: (T,T)): Unit = {

    v_W.muli(beta1).addi(gradients._1.reshape(-1,1) * (-beta1 + 1))
    val v_W_corr = v_W / (1-math.pow(beta1, t))

    s_W.muli(beta2).addi((gradients._1.reshape(-1,1) * gradients._1.reshape(-1,1)) * (-beta2 + 1))
    val s_W_corr = s_W / (1-math.pow(beta2, t))

    v_b.muli(beta1).addi(gradients._2.reshape(-1,1) * (-beta1 + 1))
    val v_b_corr = v_b / (1-math.pow(beta1, t))

    s_b.muli(beta2).addi((gradients._2.reshape(-1,1) * gradients._2.reshape(-1,1)) * (-beta2 + 1))
    val s_b_corr = s_b / (1-math.pow(beta2, t))

    W.reshape(-1,1).subi(gradients._1.reshape(-1,1) * eta * v_W_corr / (sqrt(s_W_corr) + epsilon))
    b.reshape(-1,1).subi(gradients._2.reshape(-1,1) * eta * v_b_corr / (sqrt(s_b_corr) + epsilon))

    t += 1
  }
}
