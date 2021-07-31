import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;

public class SOA_Project {
  
  public static void dataSetProcessing(DataSetRow row, List<String> images){
    images.add(row.getImage());
  } 

  public static void main(String[] args) {
    SparkConf conf = new SparkConf().setAppName("SOA Project").setMaster("yarn");
    JavaSparkContext sc = new JavaSparkContext(conf);
    SparkSession spark = SparkSession.builder().getOrCreate();

    //Creazione java bean per la definizione dello schema del DataSet
		Encoder<DataSetRow> dataSetEncoder = Encoders.bean(DataSetRow.class);
    String path = "Arienzo-Giordano-Scotti/DataSetPic.json";

    Dataset<DataSetRow> dataset = spark.read().json(path).as(dataSetEncoder);

    List<String> images = new ArrayList<String>();

    dataset.toJavaRDD().foreach(row -> dataSetProcessing(row, images));

    JavaRDD<String> result = sc.parallelize(images);

    result.saveAsTextFile("Arienzo-Giordano-Scotti/result.txt");

    //Iterazione del DataSet
    dataset.foreach((ForeachFunction<DataSetRow>) row -> System.out.println(row.getId()));

    sc.close();
    spark.stop();
  }
}
