package com.intel.analytics.bigdl.friesian

object scalatest {
  def main(args: Array[String]): Unit = {
    val matrix = for {
      r <- 0 until 138
      c <- 0 until 138
      d <- 0 until 32
    } yield r + c + d
    val m1 = matrix.toArray
    val m2 = m1.grouped(32).toArray
    val m3 = m2.grouped(138).toArray
    val m4 = Array(m3)
    println("a")
    println("b")
  }
}
