/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.orca.python

import com.intel.analytics.bigdl.orca.inference.InferenceModel
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}

import java.util.{Base64, List => JList}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.common.PythonZoo
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.ipc.ArrowStreamReader

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.api.java.{UDF1, UDF2}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{col, rand, row_number, spark_partition_id, udf, log => sqllog}

import java.io.ByteArrayInputStream
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object PythonOrca {

  def ofFloat(): PythonOrca[Float] = new PythonOrca[Float]()

  def ofDouble(): PythonOrca[Double] = new PythonOrca[Double]()
}

class PythonOrca[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def inferenceModelDistriPredict(model: InferenceModel, sc: JavaSparkContext,
                                  inputs: JavaRDD[JList[com.intel.analytics.bigdl.dllib.
                                  utils.python.api.JTensor]],
                                  inputIsTable: Boolean): JavaRDD[JList[Object]] = {
    val broadcastModel = sc.broadcast(model)
    inputs.rdd.mapPartitions(partition => {
      val localModel = broadcastModel.value
      partition.map(inputs => {
        val inputActivity = jTensorsToActivity(inputs, inputIsTable)
        val outputActivity = localModel.doPredict(inputActivity)
        activityToList(outputActivity)
      })
    })
  }

  def sdfReshape(df: DataFrame, columns: JList[String], shapes: JList[JList[Int]]): DataFrame = {
    val spark = df.sparkSession
    val columnScala = columns.asScala
    val shapeScala = shapes.asScala.map(_.asScala.toArray[Int]).toArray
    val reverseShapeArr = shapeScala.map(_.reverse.dropRight(1))

    val col_idx = columnScala.map(col_n => {
      val idx = df.columns.indexOf(col_n)
      if (idx == -1) {
        throw new IllegalArgumentException(s"The column name $col_n does not exist")
      }
      idx
    })

    val resultRDD = df.rdd.map(row => {
//      var origin = row.toSeq.toVector
//      for ((idx, shape) <- col_idx zip shapeScala) {
//        val shapeReverse = shape.reverse.dropRight(1)
//        var groupedArr: Array[Any] = row.getAs[mutable.WrappedArray[Float]](idx).toArray
//        for (s <- shapeReverse) {
//          groupedArr = groupedArr.grouped(s).toArray
//        }
//        origin = origin.updated(idx, groupedArr)
//      }
//      Row.fromSeq(origin)
      val rowList = (col_idx zip reverseShapeArr).map(idxShape => {
        var groupedArr: Array[Any] = row.getAs[mutable.WrappedArray[Float]](idxShape._1).toArray
        for (s <- idxShape._2) {
          groupedArr = groupedArr.grouped(s).toArray
        }
        groupedArr
      })
      Row.fromSeq(rowList)
    })

    val resultStruct = (columnScala zip shapeScala).map(nameShape => {
      var structType: DataType = FloatType
      for (_ <- nameShape._2.indices) {
        structType = ArrayType(structType)
      }
      StructField(nameShape._1, structType, true)
    }).toArray
    val schema = StructType(resultStruct)
    val resultDF = spark.createDataFrame(resultRDD, schema)
    resultDF
//    var resultDF = df
//    (columns zip shapeScala).foreach(cs => {
//      val colShape = cs._2
//      var structType: DataType = FloatType
//      for (_ <- colShape.indices) {
//        structType = ArrayType(structType)
//      }
////      val reshapeUDF = udf(reshape, structType)
//      val reshapeUDF = udf((data: mutable.WrappedArray[Float], shape: Array[Int]) => {
//        val shapeReverse = shape.reverse.dropRight(1)
//        var groupedArr: Array[Float] = data.toArray
//        var iter: Iterator = null
//        for (s <- shapeReverse) {
////          groupedArr = groupedArr.grouped(s)
//          iter = groupedArr.grouped(s)
//        }
//        groupedArr.asInstanceOf[Array[Array[Array[Float]]]]
//      })
//
//      val size = udf((data: mutable.WrappedArray[Float]) => {
//        data.length
//      })
////      val reshapeUDF = udf(new UDF2[mutable.WrappedArray[Float], Array[Int]] {
////  override def call(t1: mutable.WrappedArray[Float]): Array[Int] = {
////
////  }
////})
//      resultDF = resultDF.withColumn(cs._1, reshapeUDF(col(s"`${cs._1}`"), lit(colShape)))
////      resultDF = resultDF.withColumn(cs._1, reshapeUDF(col("`tf.identity`")))
//      resultDF.show()
//    })
//    resultDF
  }

  def arrowTest(inputs: JavaRDD[String], outputNames: JList[String], outShapes: JList[JList[Int]])
  : DataFrame = {
    val spark = SparkSession.builder.config(inputs.sparkContext.getConf).getOrCreate()
    val outputNamesScala = outputNames.asScala
    val outputShapesScala = outShapes.asScala.map(_.asScala.toArray[Int]).toArray
//    val outputShapesScala = Array(Array(2, 3), Array(2, 4))
    val de_rdd = inputs.rdd.flatMap(hexStr => {
      val allocator = new RootAllocator()
      val in = new ByteArrayInputStream(hexStringToByteArray(hexStr))
      val stream = new ArrowStreamReader(in, allocator)
      val vsrhex = stream.getVectorSchemaRoot
      val outputVectorReaders = outputNamesScala.map(name => {
        vsrhex.getVector(name).getReader
      })
      stream.loadNextBatch()
      val rowCount = vsrhex.getRowCount
      val batchedListResults = (0 until rowCount).map(i => {
        val row_vector = outputVectorReaders.zipWithIndex.map(readerIdxTuple => {
          val reader = readerIdxTuple._1
          val idx = readerIdxTuple._2
          reader.setPosition(i)
          val shape = outputShapesScala(idx).reverse.dropRight(1)
          val dataArr = ArrayBuffer[Float]()
          while (reader.next()) {
            val floatReader = reader.reader()
            if (floatReader.isSet) {
              dataArr += floatReader.readFloat()
            }
          }
          var groupedArr: Array[Any] = dataArr.toArray
          for (s <- shape) {
            groupedArr = groupedArr.grouped(s).toArray
          }
          groupedArr
        })
        Row.fromSeq(row_vector)
      })
      stream.close()
      in.close()
      allocator.close()
      batchedListResults
    })
//    val c = de_rdd.collect()
    // TODO: merge schema
//    var schema = StructType()
    val resultStruct = (outputNamesScala zip outputShapesScala).map(nameShape => {
      var structType: DataType = FloatType
      for (_ <- nameShape._2.indices) {
        structType = ArrayType(structType)
      }
      StructField(nameShape._1, structType, true)
    }).toArray
//    schema = StructType(schema.fields ++: resultStruct)
    val schema = StructType(resultStruct)
    val df = spark.createDataFrame(de_rdd, schema)
//    df.show(truncate=false)

    df
  }

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

  def generateStringIdx(df: DataFrame, columns: JList[String], frequencyLimit: String = null,
                        orderByFrequency: Boolean = false)
  : JList[DataFrame] = {
    var default_limit: Option[Int] = None
    val freq_map = scala.collection.mutable.Map[String, Int]()
    if (frequencyLimit != null) {
      val freq_list = frequencyLimit.split(",")
      for (fl <- freq_list) {
        val frequency_pair = fl.split(":")
        if (frequency_pair.length == 1) {
          default_limit = Some(frequency_pair(0).toInt)
        } else if (frequency_pair.length == 2) {
          freq_map += (frequency_pair(0) -> frequency_pair(1).toInt)
        }
      }
    }
    val cols = columns.asScala.toList
    cols.map(col_n => {
      val df_col = df
        .select(col_n)
        .filter(s"${col_n} is not null")
        .groupBy(col_n)
        .count()
      val df_col_ordered = if (orderByFrequency) {
        df_col.orderBy(col("count").desc)
      } else df_col
      val df_col_filtered = if (freq_map.contains(col_n)) {
        df_col_ordered.filter(s"count >= ${freq_map(col_n)}")
      } else if (default_limit.isDefined) {
        df_col_ordered.filter(s"count >= ${default_limit.get}")
      } else {
        df_col_ordered
      }

      df_col_filtered.cache()
      val count_list: Array[(Int, Int)] = df_col_filtered.rdd.mapPartitions(getPartitionSize)
        .collect().sortBy(_._1)  // further guarantee prior partitions are given smaller indices.
      val base_dict = scala.collection.mutable.Map[Int, Int]()
      var running_sum = 0
      for (count_tuple <- count_list) {
        base_dict += (count_tuple._1 -> running_sum)
        running_sum += count_tuple._2
      }
      val base_dict_bc = df_col_filtered.rdd.sparkContext.broadcast(base_dict)

      val windowSpec = Window.partitionBy("part_id").orderBy(col("count").desc)
      val df_with_part_id = df_col_filtered.withColumn("part_id", spark_partition_id())
      val df_row_number = df_with_part_id.withColumn("row_number", row_number.over(windowSpec))
      val get_label = udf((part_id: Int, row_number: Int) => {
        row_number + base_dict_bc.value.getOrElse(part_id, 0)
      })
      df_row_number
        .withColumn("id", get_label(col("part_id"), col("row_number")))
        .drop("part_id", "row_number", "count")
    }).asJava
  }

  def getPartitionSize(rows: Iterator[Row]): Iterator[(Int, Int)] = {
    if (rows.isEmpty) {
      Array[(Int, Int)]().iterator
    } else {
      val part_id = TaskContext.get().partitionId()
      Array(Tuple2(part_id, rows.size)).iterator
    }
  }
}
