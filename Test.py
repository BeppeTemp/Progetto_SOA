from pyspark.sql import SparkSession

def rowAsDict(row):
    return row.asDict()

def exploreDataSet(data):
    return data.asDict()

if __name__ == "__main__":
    # Definizione della Spark_Session
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .getOrCreate()

    # Caricamento del dataset
    df = spark.read.json("Arienzo-Giordano-Scotti/DataSetPic.json")

    print()
    d = df.select("id","image","people","version").rdd.map(exploreDataSet).take(1)
    print(d[0]["id"])
    print()

    spark.stop()


