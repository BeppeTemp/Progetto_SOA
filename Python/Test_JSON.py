from posixpath import split
from pyspark import rdd
from pyspark.sql import SparkSession
from pyspark.sql.types import Row

def exploreDataSet(df):
    df.show()

    data = df.select("id","image","people","version").rdd.map(lambda row: row.asDict()).collect()

    for d in data:
        print(d["id"])
        print(d["image"])
        print(d["version"])
        print(d["people"][0]["person_id"])

if __name__ == "__main__":
    # Definizione della Spark_Session
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .getOrCreate()

    # Caricamento del dataset
    df = spark.read.json("Arienzo-Giordano-Scotti/DataSetPic.json")
        
    print()
    exploreDataSet(df)
    print()

    spark.stop()
