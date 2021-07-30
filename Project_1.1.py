from pyspark.sql import SparkSession

if __name__ == "__main__":
    # Definizione della Spark_Session
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .getOrCreate()

    # Caricamento del dataset
    df = spark.read.json("Arienzo-Giordano-Scotti/DataSetPic.json")
        
    print()
    data = df.select("id","image","people","version").rdd.map(processoscript_final).take(1)
    print("Esecuzione completata")
    print()

    spark.stop()