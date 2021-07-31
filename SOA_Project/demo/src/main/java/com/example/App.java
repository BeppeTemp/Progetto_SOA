package com.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        //Definizione delle sessione spark
        SparkSession spark = SparkSession.builder.appName("Test_Application").getOrCreate();
        
        //Caricamento DataSet JSON
        String path = "Arienzo-Giordano-Scotti/DataSetPic.json";
    val dataset = spark.read.json(path)
    
    //Definizione schema da DataFrame finale
    val resultSchema = StructType(Array(
      StructField("image",StringType,true),
    ))
    }
}
