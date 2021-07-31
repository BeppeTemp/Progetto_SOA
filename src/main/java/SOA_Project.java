import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;

import org.apache.spark.sql.SparkSession;

import org.apache.spark.api.java.function.ForeachFunction;

public class SOA_Project {
  
  public static void dataSetProcessing(DataSetRow row){

  } 

  public static void main(String[] args) {
    SparkSession spark = SparkSession.builder().appName("SOA Project").getOrCreate();

    //Creazione java bean per la definizione dello schema del DataSet
		Encoder<DataSetRow> dataSetEncoder = Encoders.bean(DataSetRow.class);
    String path = "Arienzo-Giordano-Scotti/DataSetPic.json";
    //Lettura del DataSet
    Dataset<DataSetRow> dataset = spark.read().json(path).as(dataSetEncoder);

    //Iterazione del DataSet
    dataset.foreach((ForeachFunction<DataSetRow>) row -> System.out.println(row.getId()));

    spark.stop();
  }
}
