// Some Spark ML libraries import
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector


// Some helper functions
def distance(a:Vector, b:Vector) = math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d=>d*d).sum)

def distToCentroid(datum:Vector, model:KMeansModel) = {
	val cluster = model.predict(datum)
	val centroid = model.clusterCenters(cluster)
	distance(centroid, datum)
	}
	
def clusteringScore(data:RDD[Vector], k:Int) = {
   val kmeans = new KMeans()
   kmeans.setK(k)
   kmeans.setEpsilon(1.0e-6)
   val model = kmeans.run(data)
   data.map(datum => distToCentroid(datum,model)).mean()
}



// Read the data from the cluster - Spark RDD creation
val rawData = sc.textFile("/FileStore/tables/TrainData.csv")

// display the few rows. Next count by the last label.
rawData.take(5).foreach(println)
rawData.map(line=> line.split(',').last).countByValue().foreach(println)

// prepare labels and Data Vectors as Key value pairs.
val rawDataArray = rawData.map(line => line.split(','))
val labelsAndData = rawDataArray.map(Aline => (Aline(2),Vectors.dense(Aline(0).toDouble,Aline(1).toDouble)))
labelsAndData.take(5).foreach(println)

// Just create and display the Data point Vectors. Which is input for Training the k-Means Cluster
val data = labelsAndData.values.cache()
data.take(10).foreach(println)

//Training Data count. To check
data.count()

// Creating kmeans object with hyper parameters.
var kmeans = new KMeans()
kmeans.setK(2)
kmeans.setEpsilon(1.0e-6)

val model = kmeans.run(data)

var vecGood2 = model.clusterCenters(0)
var vecGood = model.clusterCenters(1)

//use remembered labels to display whichone went to which cluster
val clusterLabelCount = labelsAndData.map{
 case (label, datum) =>
  	val cluster = model.predict(datum)
  		(cluster,label)
  				}.countByValue

clusterLabelCount.foreach(println)

//count elements assigned to each cluster and print it out
println(" ======= clustering output (cluster | label | count ) ======")
clusterLabelCount.toSeq.sorted.foreach{
 case ((cluster,label), count) =>
 	println(f"$cluster%1s$label%18s$count%8s")
 	}

//calculate distance from the most distant element belong to the cluster and it's centroid
val distances = labelsAndData.values.map(datum => distToCentroid(datum,model))
val threshold = distances.max -1
val maxthreshold = distances.top(1).last
println("=====Defined outlier distance threshold ===")
println("Distance from cluster core and the most distant cluster element is : "+threshold)

// filter through training data points to detect outliers that are above the threshold
val anomalies = labelsAndData.filter {
 case(label, datum) =>
 	distToCentroid(datum, model) > threshold
 	}

println("=== detected anomalies (outlier) ===")
 	anomalies.take(10).foreach(println)
 	anomalies.count()

//outlier detection
val rawAnomalyData = sc.textFile("/FileStore/tables/InputData.csv")
val anomalyData = rawAnomalyData.map(line=> line.split(',').last).countByValue().foreach(println)

//prepare data for outlier detection(remove categorical Columns -> transform rest to double array and add
// final label column
val rawAnomalyDataArray = rawAnomalyData.map(line => line.split(','))
val labelsAndDataAnomalies = rawAnomalyDataArray.map(Aline => (Aline(2),Vectors.dense(Aline(0).toDouble,Aline(1).toDouble)))

println("==== Testing outlier detection with following test data ===")
labelsAndDataAnomalies.take(5).foreach(println)

// filter through test points to detect outliers that are further from cluster center
// then the threshold distance we have defined
val goodCands = labelsAndDataAnomalies.filter {
 case(label, datum) =>
    val goodClusterCenter = vecGood
 	distance(goodClusterCenter, datum) < maxthreshold
}

println("Good candidates: ")
 	goodCands.take(100).foreach(println)
 	goodCands.count()

val goodCands2 = labelsAndDataAnomalies.filter {
 case(label, datum) =>
    val goodClusterCenter = vecGood2
 	distance(goodClusterCenter, datum) < maxthreshold
}

println("Good candidates 2: ")
 	goodCands2.take(100).foreach(println)
 	goodCands2.count()

val anomalies = labelsAndDataAnomalies.filter {
 case(label, datum) =>
    distToCentroid(datum, model) > maxthreshold
}

println("Deteceted Anomalies: ")
 	anomalies.take(100).foreach(println)
 	anomalies.count()

