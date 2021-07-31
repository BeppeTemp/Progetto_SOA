import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;

public class SOA_Project {
  public static void main(String[] args) {
    SparkSession spark = SparkSession.builder().appName("SOA Project").getOrCreate();

    // Java Bean (data class) used to apply schema to JSON data
		Encoder<DataSetRow> dataSetEncoder = Encoders.bean(DataSetRow.class);

    String path = "Arienzo-Giordano-Scotti/DataSetPic.json";

    Dataset<DataSetRow> dataset = spark.read().json(path).as(dataSetEncoder);

    dataset.show();

    spark.stop();
  }
}
