package org.smurve.yainn.components

import org.smurve.yainn.T

/**
  * Convenient base class abstracting common aspects of different layers
  */
abstract class AbstractLayer extends Layer {

  var rhs: Option[Layer] = None

  def grads(x: T, dC_dy: T): Option[(T, T)] = None

  def fp(x: T): T = next.fp(func(x))

  def next: Layer = rhs.get

  def !!(other: Layer): Layer = {
    rhs = rhs.map(_ !! other).orElse(Some(other))
    this
  }

  def fbp(x: T, yb: T): BackPack = {
    val from_next = next.fbp(func(x), yb)
    BackPack(
      from_next.C,
      dC_dy(x, from_next.dC_dy),
      grads(x, from_next.dC_dy).toList ::: from_next.grads)
  }

  /**
    * Update all parameters by updating this layer from the top of the list
    * and passing the rest on to the subsequent layer
    * @param deltas a list of all update steps for all weights and biases
    */
  def update(deltas: List[(T, T)]): Unit = {}

}
