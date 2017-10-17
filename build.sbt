
name := "yainn"
version := "1.0"
scalaVersion := "2.11.11"
organization := "org.smurve"
val nd4jVersion = "0.9.1"
val dl4jVersion = "0.9.1"

libraryDependencies ++= Seq(
  "com.github.fommil.netlib" % "all" % "1.1.2",
  "org.clapper" %% "grizzled-slf4j" % "1.3.1",
  "org.nd4j" % "nd4j-native-platform" % nd4jVersion,
  "com.sksamuel.scrimage" %% "scrimage-core" % "2.1.6", // for visualization
  "org.scalatest" %% "scalatest" % "2.2.6" % "test",
  "org.nd4j" %% "nd4s" % nd4jVersion,
  "com.github.scopt" %% "scopt" % "3.5.0",
  "ch.qos.logback" % "logback-classic" % "1.2.3"
)

