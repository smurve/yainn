package org.smurve.yainn.data

import java.io.FileInputStream
import java.nio.file.FileSystemException

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * Loads MNIST Files for training and test into INDArrays, such that they can be processed with maximum efficiency
  * through platform specific linear algebra libraries utilizing CPU or GPU (Nvidia, if available, and CUDA installed)
  * Please see http://yann.lecun.com/exdb/mnist/ to download the files and learn more about their inner structure.
  * The aforementioned files are expected to reside in a directory given by the basepath parameter
  * @param basePath the location of the MNIST data files in the local file system.
  */
class MNISTFileLoader(basePath: String){

  private val IMG_SIZE = 28 * 28
  private val IMG_HEADER_SIZE = 16
  private val LBL_HEADER_SIZE = 8


  def readImages(fileName: String, nRecords: Int ): INDArray = {

    val headerBuffer = new Array[Byte](IMG_HEADER_SIZE)
    val buffer = new Array[Byte](nRecords * IMG_SIZE )

    val stream = new FileInputStream(s"$basePath/$fileName")
    val nh = stream.read(headerBuffer)
    if ( nh != IMG_HEADER_SIZE )
      throw new FileSystemException("Failed to read image header")

    val nb = stream.read(buffer)
    if ( nb != buffer.length )
      throw new FileSystemException("Failed to read images")

    Nd4j.create(buffer.map(p => (p & 0XFF) / 256f )).reshape(nRecords, IMG_SIZE).T
  }


  def readLabels(fileName: String, nRecords: Int ): INDArray = {

    val headerBuffer = new Array[Byte](LBL_HEADER_SIZE)
    val buffer = new Array[Byte](nRecords )
    val lblBuffer10 = new Array[Float](nRecords * 10 )

    val stream = new FileInputStream(s"$basePath/$fileName")
    val nh = stream.read(headerBuffer)
    if ( nh != LBL_HEADER_SIZE )
      throw new FileSystemException("Failed to read label header")

    val nb = stream.read(buffer)
    if ( nb != buffer.length )
      throw new FileSystemException("Failed to read labels")

    for ( i <- 0 until nRecords ) {
      lblBuffer10(10 * i + buffer(i) % 0xFF) = 1f
    }

    Nd4j.create(lblBuffer10).reshape(nRecords, 10).T
  }


}
