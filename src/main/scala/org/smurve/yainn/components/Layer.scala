package org.smurve.yainn.components

import org.smurve.yainn.T

/**
  * The essential abstraction of this implementation. In mathematical terms,
  * a layer represents a function f that maps an n-dimensional vector space into another m-dimensional vector space.
  */
trait Layer {

  /**
    *
    * @param x input value
    * @return the function value for the given x
    */
  def func(x: T): T

  /**
    * Forward pass through the entire network. <br>
    *   If this layer represents a function y=f(x) and the next and last layer a function g(y),
    *   then the forward pass represents g(f(x))
    * @param x the input value
    * @return result of applying all chained layers' functions to x, one after the other
    */
  def fp(x: T): T

  /**
    * Forward and backward pass in one go
    * @param x the input value
    * @param yb y_bar, the given true classification (label) for the input value x
    * @return a structure holding the backprop artifacts
    */
  def fbp(x: T, yb: T): BackPack

  /**
    * @throws UnsupportedOperationException if there's no next layer
    * @return the subsequent layer in this network.
    */
  @throws[UnsupportedOperationException]
  def next: Layer

  /**
    * The "chaining" operator. Chains layers to form a neural network
    * @param rhs the subsequent layer in the network
    * @throws UnsupportedOperationException if there's no next layer
    * @return this layer. Allows multiple !! operators in a row
    */
  @throws[UnsupportedOperationException]
  def !!(rhs: Layer): Layer

  /**
    * cost derivative with respect to this layer's input. Note that this method essentially implements the chain rule
    * @param x this layer's input
    * @param dC_dy_from_next cost derivative with respect to the next layer's input
    * @return The cost's total derivative up until this layer
    */
  def dC_dy(x: T, dC_dy_from_next: T): T

  /**
    * @param x the input to this layer
    * @param dC_dy_from_next the subsequent layer's total derivative of the cost function
    * @return a list of all gradients from subsequent layers with earlier layers first.
    */
  def grads(x: T, dC_dy_from_next: T): Option[(T, T)]

  /**
    * Update all parameters
    * @param grads a list of gradients for all layers, top to bottom
    */
  def update(grads: List[(T, T)]): Unit

}