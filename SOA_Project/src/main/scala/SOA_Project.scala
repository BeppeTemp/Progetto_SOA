import org.apache.spark.sql.SparkSession

object SOA_Project {
  def dataSetProcessing (){
    println("ciao")
  }

  def main(args: Array[String]) {
    //Definizione delle sessione spark
    val spark = SparkSession.builder.appName("Test Application").getOrCreate()

    //Caricamento DataSet JSON
    val path = "Arienzo-Giordano-Scotti/DataSet.json"
    val dataset = spark.read.json(path)

    println()
    println("Inizio output programma: ")
    println()

    println("ciao")

    println()
    println("Fine output programma:")
    println()

    spark.stop()
  }
}