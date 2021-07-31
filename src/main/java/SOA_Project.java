import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class SOA_Project {
  
  public static void dataSetProcessing(DataSetRow row, List<ResultRow> images){
    images.add(new ResultRow(row.getImage()));
  } 

  public static void main(String[] args) {
    SparkConf conf = new SparkConf().setAppName("SOA Project").setMaster("yarn");
    JavaSparkContext sc = new JavaSparkContext(conf);
    SparkSession spark = SparkSession.builder().getOrCreate();

    //Creazione java bean per la definizione dello schema del DataSet
		Encoder<DataSetRow> dataSetEncoder = Encoders.bean(DataSetRow.class);
    Encoder<ResultRow> resultEncoder = Encoders.bean(ResultRow.class);
    String path = "Arienzo-Giordano-Scotti/DataSet.json";

    Dataset<DataSetRow> dataset = spark.read().json(path).as(dataSetEncoder);

    List<ResultRow> images = new ArrayList<ResultRow>();

    dataset.toJavaRDD().collect().forEach(row -> images.add(new ResultRow(row.getImage())));

    //images.add(new ResultRow("ciaoooooo c'è nessuno ?"));
    //images.add(new ResultRow(dataset.first().getImage()));

    Dataset<ResultRow> result = spark.createDataset(images, resultEncoder);

    result.repartition(1).write().format("json").save("Arienzo-Giordano-Scotti/result.json");

    sc.close();
    spark.stop();
  }
}
