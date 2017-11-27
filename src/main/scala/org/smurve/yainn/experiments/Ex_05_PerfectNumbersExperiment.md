## Experiment 5: Perfect Numbers

The Scala Code: [`Ex_05_PerfectNumbersExperiment`](Ex_05_PerfectNumbersExperiment.scala)

As a side effect, our [`Layer`](../components/Layer.scala) design allows the backpropagation to continue beyond the
input layer. The gradient that is returned by the input layer tells us how we can *improve* any image to get a given 
high-accuracy classification for it. We expect the result to be somewhat close to a recognizable digit. 
While that is true for some of the digits, it isn't for all. While the perfect digits 2 and 3 are pretty much recognizable,
digits 4, 7, and 9 indicate that the network just requires some very few pixels to conclude the classification with high
accuracy

Below, you can see how the gradient of the cost function with regards to the input image dC/dx (`dC_dx`) is used to 
optimize the image 

```
      /** just some white noise image */
      var digit = Nd4j.rand(seed, 784).T / 100
    
      /** pre-define the desired classification */
      val yb = t(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
      yb(i) = 1.0
    
      /** fit the white noise to produce a particular classification */
      for  {
        _ <- 1 to nmax if euc(nn.fp(digit), yb) > 1e-2
      } {
        //val c = euc(nn.fp(digit), yb)
        //println(s"Cost: $c")
        
        // fwd-bwd pass to retrieve dC_dy (which is dC_dx here...)
        val dC_dx = nn.fbp(digit, yb, digit, update = false).dC_dy 
        digit = relu(digit - dC_dx * eta)// gradient descent: make sure the pixel values stay positive
      }
      val scale = 1.0 / (digit.maxNumber().doubleValue()+1e-6)
    
      /** see that the network recognizes its own piece of art */
      predict(nn.fp(digit*scale), (digit * scale, yb))
```