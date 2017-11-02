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

  def !!(nextLayer: Layer): Layer = {
    rhs = rhs.map(_ !! nextLayer).orElse(Some(nextLayer))
    this
  }

  def ::(previous: Layer): Layer = {
    previous.append(this)
  }

  def append( subs: Layer ): Layer = {
    rhs = Some(subs)
    this
  }

  def fbp(x: T, yb: T, orig_x: T, update: Boolean): BackPack = {
    val from_next = next.fbp(func(x), yb, orig_x, update)
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
