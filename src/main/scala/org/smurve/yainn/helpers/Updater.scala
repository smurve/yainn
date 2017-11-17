package org.smurve.yainn.helpers

import org.smurve.yainn.T

trait Updater {

  def update( W: T, b: T, gradients: (T,T)): Unit
}
