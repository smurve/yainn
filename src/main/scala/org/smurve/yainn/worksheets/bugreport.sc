import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

val x = Nd4j.zeros(5,2)

val x1 = Nd4j.create(Array(1.0, 2, 3, 4, 5)).T
val x2 = Nd4j.create(Array(5,4,3,2,1.0)).T

x(->,0) = x1
x(->,1) = x2

val xx1 = x + 1
val xx = x + 0

val x1p = xx(->,0)
val one = Nd4j.ones(1)

// this is flawed!
Nd4j.vstack(one, x1p)

// this is the workaround
Nd4j.hstack(one, x1p.T).T


val y = Nd4j.create(Array(1.0, 2,3,4,5,5,4,3,2,1)).reshape(5,2)

val y1 = Nd4j.create(Array(1.0, 2, 3, 4, 5)).T
val y2 = Nd4j.create(Array(5,4,3,2,1.0)).T

//y(->,0) = y1
//y(->,1) = y2

val y1p = y(->,0)

Nd4j.vstack(one, y1p)

