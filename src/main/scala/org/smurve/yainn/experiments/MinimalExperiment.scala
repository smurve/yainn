package org.smurve.yainn.experiments

import org.nd4j.linalg.factory.Nd4j
import org.smurve.yainn._
import org.smurve.yainn.components.{Affine, BackPack, Layer, Output}
import org.smurve.yainn.data.MinimalDataIterator
import org.nd4s.Implicits._

object MinimalExperiment {

  def main(args: Array[String]): Unit = {
    val seed = 1234L

    val NOISE = 0.7
    val BATCH_SIZE = 20
    val NUM_BATCHES = 20
    val ETA = .3
    val N_TEST = 100

    val data = new MinimalDataIterator(NOISE, BATCH_SIZE, seed)
    val testSet = data.nextBatch(N_TEST)

    val nn = createNetwork (seed)

    for ( b <- 1 to NUM_BATCHES ) {

      val trainingSet = data.nextBatch()

      val allCost = for ((symbol, label) <- trainingSet) yield {

        val BackPack(cost, _, grads) = nn.fbp(symbol, label)
        val deltas = grads.map(p => {
          (p._1 * ETA, p._2 * ETA)
        })
        nn.update(deltas)
        cost
      }
      val avg = allCost.sum / BATCH_SIZE

      println(s"Batch Nr: $b: Cost=$avg")
      //println("Validating...")
      val successRate = validate ( nn, testSet ) * 100
      println(s"Sucess rate: $successRate")
    }


  }

  /**
    * create the network from randomly initialized weights and biases
    * @param seed a random seed
    * @return the first layer representing the network
    */
  def createNetwork (seed: Long): Layer = {
    val (nx, nh, ny) = (9, 16, 2)

    val W0 = Nd4j.rand(nh, nx, seed) - 0.5
    val b0 = Nd4j.rand(nh, 1, seed) - 0.5
    val W1 = Nd4j.rand(ny, nh, seed) - 0.5
    val b1 = Nd4j.rand(ny, 1, seed) - 0.5

    Affine(W0, b0) !! Sigmoid() !! Affine(W1, b1) !! Sigmoid() !! Output(euc, euc_prime)
  }

  /**
    * Count the number of true positives within a test set.
    */
  def validate(nn: Layer, testSet: List[(T, T)]): Double =
    testSet.map{ case (image, label) => if ( equiv(nn.fp(image), label)) 1 else 0 }.sum.toDouble / testSet.size


}

