package com.kamali.exe
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.hadoop.fs.s3a.S3AFileSystem 
import org.apache.hadoop._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.sql._
import org.apache.spark._
import org.apache.log4j._
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.functions.count
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
object TaxiLogistic{
 

  def main(args: Array[String]) {
   
  
   //sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId","AKIAJY3KKR5VZK4VKCPQ")
   //sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey","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")
 
  
     
  val spark=SparkSession.builder().master("local").appName("StreamingTaxi").getOrCreate()
  import spark.implicits._
  
   
  spark.sparkContext.setLogLevel("WARN")
 val OurSchema=StructType(Array(
 StructField("vendorID",IntegerType,true),
 StructField("pickupDatetime",TimestampType,true),
 StructField("dropDateTime",TimestampType,true),
 StructField("NoOfPass",IntegerType,true),
 StructField("tripDistance",DoubleType,true),
 StructField("RatecodeID",IntegerType,true),
 StructField("StoreFwdFlag",StringType,true),
 StructField("PULocationID",IntegerType,true),
 StructField("DOLocationID",IntegerType,true),
 StructField("PaymentType",IntegerType,true),
 StructField("fare",DoubleType,true),
 StructField("extra",DoubleType,true),
 StructField("mtaTax",DoubleType,true),
 StructField("tip",DoubleType,true),
 StructField("tolls",DoubleType,true),
 StructField("improveCharge",DoubleType,true),
 StructField("totalAmount",DoubleType,true)
  ))

  
  //val df=spark.read.option("header", true).schema(OurSchema).csv("s3n://s3.amazonaws.com/streamtaxidata/*.csv")
  val df=spark.read.option("header", true).schema(OurSchema).csv("/home/kiran/Downloads/yellow_tripdata_2017-01.csv") 
 // val df = spark.read.option("header", true).schema(OurSchema).csv("/home/kiran/Downloads/yellow_tripdata_2017-01.csv")  
 // val data=spark.read.textFile("/home/kiran/Downloads/yellow_tripdata_2017-01.csv")
  /*val header=data.first()
  val Data2=data.filter(row=>row!=header)
  val header2=Data2.first()
  val FilteredData=Data2.filter(row=>row!=header2)
  
    val pickups=FilteredData.rdd.map{ line=>
    val tempList=line.split(",").toArray
    var PULocationID=tempList.apply(7).toInt
    val k =Vectors.dense((PULocationID))
    (k)
    }
  pickups.cache()
  */
   df.show()
   df.printSchema()
   df.createOrReplaceTempView("DataTable")

  
  
  val picksAnddrops= spark.sql("SELECT PULocationID,DOLocationID FROM DataTable WHERE PULocationID!=DOLocationID")
  picksAnddrops.show()
  //pickups.take(5).foreach(println)
val smallSchema=StructType(Array(StructField("SourceDestinations",StringType,true)))
  val tuples=picksAnddrops.rdd.map{ line=>
           val p =line.apply(0)
           val d=line.apply(1)
           val a=(p,d).toString()
           Row(a) 
         }
  val sTuples=spark.createDataFrame(tuples,smallSchema)
sTuples.show()
val count=sTuples.groupBy("SourceDestinations").count().orderBy(desc("count"))
count.show()
count.createOrReplaceTempView("counts")
val mostFrequentPickedLocation=spark.sql("SELECT First(SourceDestinations) FROM counts")
mostFrequentPickedLocation.show()
val picked=mostFrequentPickedLocation.rdd.take(1)
val item=picked.apply(0)
val k=item.toString().split(",").take(1)
val a=k.apply(0).toString()
val z=a.splitAt(2)
val top=z._2
val pickTimes=spark.sql(s"SELECT pickupDateTime FROM DataTable WHERE PULocationID=$top")
//pickTimes.show()
val numClusters = 4
val numIterations = 20
//val clusters = KMeans.train(pickups, numClusters, numIterations)
//clusters.save(spark.sparkContext,"/home/kiran/Kmeans")
//predicting price 

val selectivePriceData=df.select(df("totalAmount").as("label"),$"tripDistance",$"RatecodeId",$"PULocationID",$"DOLocationID",
    $"fare",$"extra",$"mtaTax",$"tip",$"tolls",$"improveCharge")

selectivePriceData.show()    
val assembler = new VectorAssembler().setInputCols(Array("tripDistance","RatecodeId",
   "PULocationID","DOLocationID","fare","extra","mtaTax","tip","tolls","improveCharge")).setOutputCol("features")
  
val TrainingPriceData = assembler.transform(selectivePriceData).select($"label",$"features")
TrainingPriceData.cache()
TrainingPriceData.show()
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
val lrModel = lr.fit(TrainingPriceData)
lrModel.save("/home/kiran/LogisticRegression")

val sameModel = LogisticRegression.load("/home/kiran/LinearRegression")

val df2 = spark.read.option("header", true).schema(OurSchema).csv("/home/kiran/Downloads/yellow_tripdata_2017-02.csv")
//val df2=spark.read.option("header", true).schema(OurSchema).csv("s3n://s3.amazonaws.com/streamtaxidata/*.csv")

df2.show()
val testPriceData=df2.select($"tripDistance",$"RatecodeId",$"PULocationID",$"DOLocationID",$"fare",$"extra",$"mtaTax",$"tip",$"tolls",$"improveCharge")
testPriceData.show()
val TestData = assembler.transform(testPriceData).select($"features")
TestData.show()
val Pricepredictions=lrModel.transform(TestData)
Pricepredictions.show()
//predicting tip
val selectiveTipData=df.select(df("tip").as("label"),$"tripDistance",$"RatecodeId",$"PULocationID",$"DOLocationID",
    $"fare",$"extra",$"mtaTax",$"tolls",$"improveCharge")
    
selectiveTipData.show()
val TipAssembler = new VectorAssembler().setInputCols(Array("tripDistance","RatecodeId",
   "PULocationID","DOLocationID","fare","extra","mtaTax","tolls","improveCharge")).setOutputCol("features")
val TrainingTipData = TipAssembler.transform(selectiveTipData).select($"label",$"features")
TrainingTipData.cache()
TrainingTipData.show()   
val lrModel2 = lr.fit(TrainingTipData)   
lrModel2.save("/home/kiran/LinearRegression")

val sameModel2 = LogisticRegression.load("/home/kiran/LinearRegression")

val testTipData=df2.select($"tripDistance",$"RatecodeId",$"PULocationID",$"DOLocationID",$"fare",$"extra",$"mtaTax",$"tip",$"tolls",$"improveCharge")
testTipData.show()
val TestTipData = TipAssembler.transform(testTipData).select($"features")
TestTipData.show()

val Tippredictions=lrModel2.transform(TestTipData)
Tippredictions.show()

import org.apache.spark.mllib.evaluation.MulticlassMetrics

val realTips=df2.select($"tip")
val predictedTips=Tippredictions.select($"prediction")
val output=realTips.union(predictedTips)


  }
}