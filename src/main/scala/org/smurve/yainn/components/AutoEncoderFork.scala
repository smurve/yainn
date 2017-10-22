package org.smurve.yainn.components
import org.smurve.yainn.T
import org.nd4s.Implicits._

/**
  * fork to allow for a branch to complete the network up to here as an autoencoder
  * Only supports auto-updating components on the ae branch, since it can't propagate back a pair of gradients.
  * @param ae the tail part of the ae
  * @param lambda: the percentage that the ae contributes to the cost function
  */
case class AutoEncoderFork ( ae: Layer, lambda: Double ) extends AbstractLayer {
  require(lambda >= 0 && lambda <= 1 )
  /**
    * @param x input value
    * @return x: this fork doesn't do a thing
    */
  override def func(x: T): T = throw new UnsupportedOperationException("Not used here.")

  /**
    * Forward pass through the actual network, not the ae tail. <br>
    * If this layer represents a function y=f(x) and the next and last layer a function g(y),
    * then the forward pass represents g(f(x))
    *
    * @param x the input value
    * @return result of applying all chained layers' functions to x, one after the other
    */
  override def fp(x: T): T = next.fp(x)

  /**
    * Forward and backward pass in one go
    *
    * @param x  the input value
    * @param yb y_bar, the given true classification (label) for the input value x
    * @return a structure holding the backprop artifacts
    */
  override def fbp(x: T, yb: T, orig_x: T): BackPack = {

    val from_ae = ae.fbp(x, orig_x, orig_x)
    val from_nn = next.fbp(x, yb, orig_x)
    val aec = lambda * from_ae.C
    var nnc = ( 1 - lambda ) * from_nn.C
    val totalCost = (1-lambda) * nnc + lambda * aec
    val total_dC_dy = from_nn.dC_dy * ( 1 - lambda ) + from_ae.dC_dy * lambda
    BackPack(totalCost, total_dC_dy, from_nn.grads)
  }

  /**
    * cost derivative with respect to this layer's input. Note that this method essentially implements the chain rule
    *
    * @param x               this layer's input
    * @param dC_dy_from_next cost derivative with respect to the next layer's input
    * @return The cost's total derivative up until this layer
    */
  override def dC_dy(x: T, dC_dy_from_next: T): T =
    throw new UnsupportedOperationException("Not used here.")

  /**
    * @param x               the input to this layer
    * @param dC_dy_from_next the subsequent layer's total derivative of the cost function
    * @return a list of all gradients from subsequent layers with earlier layers first.
    */
  override def grads(x: T, dC_dy_from_next: T): Option[(T, T)] =
    throw new UnsupportedOperationException("Not supporting updatable parameters.")

  /**
    * Update all parameters
    *
    * @param deltas a list of gradients for all layers, top to bottom
    */
  override def update(deltas: List[(T, T)]): Unit = {
    ae.update(deltas)
    next.update(deltas)
  }
}
