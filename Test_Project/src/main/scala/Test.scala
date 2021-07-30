import org.apache.spark.sql.SparkSession

object Test {
  def main(args: Array[String]) {
    val logFile = "Arienzo-Giordano-Scotti/DataSetPic.json" // Should be some file on your system
    val spark = SparkSession.builder.appName("Test Application").getOrCreate()
    val logData = spark.read.textFile(logFile).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println(s"Lines with a: $numAs, Lines with b: $numBs")
    spark.stop()
  }
}