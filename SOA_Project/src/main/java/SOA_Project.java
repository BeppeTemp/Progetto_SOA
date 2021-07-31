import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;

public class SOA_Project {
  public static void main(String[] args) {
    SparkSession spark = SparkSession.builder().appName("Simple Application").getOrCreate();


    System.out.println("finito");

    spark.stop();
  }
}
