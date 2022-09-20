package com.intel.analytics.bigdl.friesian

import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.complex.ListVector
import org.apache.arrow.vector.{IntVector, VectorSchemaRoot}
import org.apache.arrow.vector.ipc.{ArrowFileReader, ArrowStreamReader}
import org.apache.spark.sql.SparkSession
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.sql.util.ArrowUtils

import java.io.{ByteArrayInputStream, File, FileInputStream}
import java.util
import javax.xml.bind.DatatypeConverter
import scala.io.Source
import com.intel.analytics.bigdl.orca.python.PythonOrca

import scala.collection.JavaConverters._

object scalatest {



  def hexStringToByteArray(s: String): Array[Byte] = {
    val len = s.length
    val data = new Array[Byte](len / 2)
    var i = 0
    while ( {
      i < len
    }) {
      data(i / 2) = ((Character.digit(s.charAt(i), 16) << 4) + Character.digit(s.charAt(i + 1), 16)).toByte
      i += 2
    }
    data
  }

  def main(args: Array[String]): Unit = {
//    val matrix = for {
//      r <- 0 until 138
//      c <- 0 until 138
//      d <- 0 until 32
//    } yield r + c + d
//    val m1 = matrix.toArray
//    val m2 = m1.grouped(32).toArray
//    val m3 = m2.grouped(138).toArray
//    val m4 = Array(m3)
//    println("a")
//    println("b")

    val spark: SparkSession = SparkSession.builder()
      .master("local[1]").appName("SparkByExamples.com")
      .getOrCreate()

    val hexFile = "/home/yina/Documents/BigDL/python/orca/example/learn/openvino/hex"
//    for (line <- Source.fromFile(hexFile).getLines()) {
//      println(line)
//    }
    val line = Source.fromFile(hexFile).getLines().next()
    val rdd = spark.sparkContext.parallelize(Array(line, line), numSlices = 2)
    val orca = PythonOrca.ofFloat()
    val shapes = Array(Array(2, 3), Array(2, 4))
    val jShape = shapes.map(_.toList.asJava).toList.asJava
    val result = orca.arrowTest(JavaRDD.fromRDD(rdd), Array("a", "b").toList.asJava, jShape)



//    val allocator = new RootAllocator()
//    try {
//
//      val in = new ByteArrayInputStream(hexStringToByteArray(line))
//      val stream = new ArrowStreamReader(in, allocator)
//      val vsrhex = stream.getVectorSchemaRoot
//      val hex_a = vsrhex.getVector("a")
//      stream.loadNextBatch()
//      val vsr2 = stream.getVectorSchemaRoot
//      val fileInputStream = new FileInputStream(new File("/home/yina/Documents/BigDL/python/orca/example/learn/openvino/test.arrow"))
//      val reader = new ArrowFileReader(fileInputStream.getChannel, allocator)
//      val vsr = stream.getVectorSchemaRoot
//      System.out.println("Record batches in file: " + reader.getRecordBlocks.size)
//
//      reader.getRecordBlocks.forEach(arrowBlock => {
//        reader.loadRecordBatch(arrowBlock)
//        val root = reader.getVectorSchemaRoot
//        val vecs = root.getFieldVectors
//        val a_vec = root.getVector("a")
//        val listReader = a_vec.getReader
//        for (i <- 0 until a_vec.getValueCount) {
//          listReader.setPosition(i)
//          while (listReader.next()) {
//            val intReader = listReader.reader()
//            if (intReader.isSet) {
//              print(intReader.readInteger())
//              print(" ")
//            }
//          }
//          println()
//        }
//        val a_1 = a_vec.getObject(0).asInstanceOf[util.ArrayList[Integer]]
//        val dataArray = new Array[Float](a_vec.getValueCount)
//        dataArray.indices.foreach(i => dataArray(i) = a_vec.get(i))
//        val b_vec = root.getVector("b")
//        val b = new Array[Float](b_vec.getValueCount)
//        b.indices.foreach(i => b(i) = a_vec.get(i))
//        System.out.println("VectorSchemaRoot read: \n" + root.contentToTSVString)
//      })
//      try {
//        reader.loadNextBatch
//        val readRoot = reader.getVectorSchemaRoot
//        // get the encoded vector
//        val intVector = readRoot.getVector(0).asInstanceOf[Nothing]
//      } finally if (reader != null) reader.close()
//    }
  }
}
