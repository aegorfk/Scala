//Decision Trees
val datadir = "/home/azureuser/data"

//Create a SQL Context from Spark context
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._

//Load the CSV file into a RDD
val bankData = sc.textFile(datadir + "/bank.csv")
bankData.cache()
bankData.count()

//Remove the first line (contains headers)
val firstLine=bankData.first()
val dataLines = bankData.filter(x => x != firstLine)
dataLines.count()

//Convert the RDD into a Dense Vector
//Change labels to numeric ones
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

def transformToNumeric( inputStr : String) : Vector = {
  val attList=inputStr.split(";")

  val age=attList(0).toFloat
  //convert outcome to float
  val outcome = attList(16).contains("no") match {
    case  true => 1.0
    case  false    => 0.0
  }

  //create indicator variables for single/married
  val single = attList(2).contains("single") match {
    case  true => 1.0
    case  false    => 0.0
  }
  val married = attList(2).contains("married") match {
    case  true => 1.0
    case  false    => 0.0
  }
  val divorced = attList(2).contains("divorced") match {
    case  true => 1.0
    case  false    => 0.0
  }

  //create indicator variables for education
  val primary = attList(3).contains("primary") match {
    case  true => 1.0
    case  false    => 0.0
  }
  val secondary = attList(3).contains("secondary") match {
    case  true => 1.0
    case  false    => 0.0
  }
  val tertiary = attList(3).contains("tertiary") match {
    case  true => 1.0
    case  false    => 0.0
  }

  //convert default to float
  val default = attList(4).contains("no") match {
    case  true => 1.0
    case  false    => 0.0
  }
  //convert balance amount to float
  val balance = attList(5).contains("no") match {
    case  true => 1.0
    case  false    => 0.0
  }
  //convert loan to float
  val loan = attList(7).contains("no") match {
    case  true => 1.0
    case  false    => 0.0
  }
  //Filter out columns not wanted at this stage
  val values= Vectors.dense(outcome, age, single, married,
    divorced, primary, secondary, tertiary,
    default, balance, loan )
  return values
}
//Change to a Vector
val bankVectors = dataLines.map(transformToNumeric)
bankVectors.cache()
bankVectors.collect()

//Statistical Analysis
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
val bankStats=Statistics.colStats(bankVectors)
bankStats.min
bankStats.max
val colMeans=bankStats.mean
val colVariance=bankStats.variance
val colStdDev=colVariance.toArray.map( x => Math.sqrt(x))

Statistics.corr(bankVectors)

//Transform to a Data Frame
//Drop columns with low correlation

def transformToLabelVectors(inStr : Vector  ) : (Float,Vector) = {
  val values = ( inStr(0).toFloat,
    Vectors.dense(inStr(1),inStr(2),inStr(3),
      inStr(4),inStr(5),inStr(6),inStr(7),
      inStr(8),inStr(9),inStr(10)))
  return values
}

val bankLp = bankVectors.map(transformToLabelVectors)
bankLp.collect()
val bankDF = sqlContext.createDataFrame(bankLp).toDF("label","features")
bankDF.select("label","features").show(10)

   
//Indexing needed as pre-req for Decision Trees
import org.apache.spark.ml.feature.StringIndexer
val stringIndexer = new StringIndexer()
stringIndexer.setInputCol("label")
stringIndexer.setOutputCol("indexed")
val si_model = stringIndexer.fit(bankDF)
val indexedBank = si_model.transform(bankDF)
indexedBank.select("label","indexed","features").show()
indexedBank.groupBy("label","indexed").count().show()


//Split into training and testing data
val Array(trainingData, testData) = indexedBank.randomSplit(Array(0.9, 0.1))
trainingData.count()
testData.count()

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

//Create the model
val dtClassifier = new DecisionTreeClassifier()
dtClassifier.setMaxDepth(4)
dtClassifier.setLabelCol("indexed")
val dtModel = dtClassifier.fit(trainingData)

dtModel.numNodes
dtModel.depth

//Predict on the test data
val predictions = dtModel.transform(testData)
predictions.select("prediction","indexed","label","features").show()

val evaluator = new MulticlassClassificationEvaluator()
evaluator.setPredictionCol("prediction")
evaluator.setLabelCol("indexed")
evaluator.setMetricName("precision")
evaluator.evaluate(predictions)      

//Draw a confusion matrix
predictions.groupBy("indexed","prediction").count().show()

