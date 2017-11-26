# YAINN
## Yet Another Introduction to Neural Networks

*documentation still under construction, code is pretty much ready*

Welcome the source code of my blog "Yet another introduction to neural networks"!

Since this is a scala project, you'll find all sources in src/main/scala. 

The main package is [org.smurve.yainn](src/main/scala/org/smurve/yainn), and within there 
you'll find the following packages

 - [components](src/main/scala/org/smurve/yainn/components): the various functional layers
 - [data](src/main/scala/org/smurve/yainn/data): the iterators that provide the input values (images)
 - [experiments](src/main/scala/org/smurve/yainn/experiments/Overview.md): executable classes demonstrating the learning success
 - [worksheets](src/main/scala/org/smurve/yainn/worksheets): some interactive worksheets to play with
 
And you may be surprised to find some (not too many, though) [tests](src/test/scala/org/smurve/yainn) in this project. 
 
###  Install linear algebra libraries
In order to have sufficient performance on your platform, I recommend you install a 
linear algebra accelerator - such as [OpenBLAS](http://www.openblas.net/) - before moving on. If you are a very patient
person and like to have a lot of time for coffee or alike, just go ahead. I'll work, anyway.
 
### Experiments are variants, not versions
The idea of experiments