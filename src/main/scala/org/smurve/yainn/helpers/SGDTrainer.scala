package org.smurve.yainn.helpers

import org.nd4s.Implicits._
import org.smurve.yainn.components.{BackPack, Layer}
import org.smurve.yainn.data.DataIterator
import org.smurve.yainn.experiments.HiddenLayersMNISTExperiment.{printstats, successCount}
import org.smurve.yainn.experiments.Params
import org.smurve.yainn.timed

/**
  * SGD trainer that can train a network with more than one entry layer
  * Different entry layers may represent different preprocessers, like shift, skew, scale that multiply the input data
  *
  * @param nns a list of entry layers to feed, the first is going to be used for testing
  */
class SGDTrainer(nns: List[Layer]) {

  def train(iterator: DataIterator, params: Params) {

    val testSet = iterator.newTestData(params.TEST_SIZE)

    for (e <- 1 to params.NUM_EPOCHS) { // for the given number of epochs
      var duration = 0L

      while (iterator.hasNext) { // for all mini-batches
        val (trainingImages, trainingLabels) = iterator.nextMiniBatch()

        for (nn <- nns) {
          /** forward-backward pass through the entire network, the entire batch in a single go, and measure time */
          val (res, time) = timed(nn.fbp(trainingImages, trainingLabels, trainingImages))
          val BackPack(cost, _, grads) = res
          duration += time //

          /** multiply all gradients by a common learning rate and update the gradients with it */
          val deltas = grads.map({ case (grad_bias, grad_weight) =>
            (grad_bias * params.ETA, grad_weight * params.ETA)
          })
          nn.update(deltas)
          if (!iterator.hasNext) {
            /** Do some stats at the end of each epoch */
            println(s"Epoch Nr. $e, after ${params.NUM_BATCHES} batches: Cost=$cost")
            printstats(deltas)
          }
        }
      }
      iterator.reset()

      /** restart from the first mini-batch */

      /** validate the networks performance with the help of the test set */
      val successRate = successCount(nns.head, testSet).sumT[Double] * 100.0 / params.TEST_SIZE
      println(s"Net learning time: $duration ms, Sucess rate: $successRate")
    }
  }

}
