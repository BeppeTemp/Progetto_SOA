import java.util.List;

import javax.xml.crypto.Data;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import org.apache.spark.api.java.function.ForeachFunction;

public class SOA_Project {
  public static void main(String[] args) {
    SparkSession spark = SparkSession.builder().appName("SOA Project").getOrCreate();

    // Java Bean (data class) used to apply schema to JSON data
		Encoder<DataSetRow> dataSetEncoder = Encoders.bean(DataSetRow.class);

    String path = "Arienzo-Giordano-Scotti/DataSetPic.json";

    Dataset<DataSetRow> dataset = spark.read().json(path).as(dataSetEncoder);

    dataset.foreach((ForeachFunction<DataSetRow>) row -> System.out.println(row));

    spark.stop();
  }
}
