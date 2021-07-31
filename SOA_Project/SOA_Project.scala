import org.apache.spark.sql.types.{IntegerType,StringType,StructType,StructField}
import org.apache.spark.sql._
import java.io._

object SOA_Project {
  def dataSetProcessing (row: Row, result: DataFrame, spark: SparkSession, resultSchema: StructType){
    val newRow = List[Row](row.getAs("image"))



    val rddObj = spark.sparkContext.parallelize(newRow)
    result.union(rddObj)
  }

  def main(args: Array[String]) {
    //Definizione delle sessione spark
    val spark = SparkSession.builder.appName("Test_Application").getOrCreate()

    //Caricamento DataSet JSON
    val path = "Arienzo-Giordano-Scotti/DataSetPic.json"
    val dataset = spark.read.json(path)
    
    //Definizione schema da DataFrame finale
    val resultSchema = StructType(Array(
      StructField("image",StringType,true),
    ))

    //Creazione dal DataFrame vuoto
    val result = spark.createDataFrame(spark.sparkContext.emptyRDD[Row],resultSchema)

    println()
    println("Inizio output programma: ")
    println()

    //Iterazione del DataFrame caricato
    dataset.foreach(row => dataSetProcessing(row, result, spark, resultSchema))

    //Scrittura del DataFrame finale su HDFS
    result.write.json("Arienzo-Giordano-Scotti/result.json")

    println()
    println("Fine output programma:")
    println()

    spark.stop()
  }
}