package org.apache.spark.ml.made

import breeze.linalg.InjectNumericOps
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait HasYCol extends Params {
  final val yCol: Param[String] = new Param[String](this, "yCol", "y column name")

  final def getYCol: String = $(yCol)

  def setYCol(value: String): this.type = set(yCol, value)
}

trait LinearRegressionParams extends HasInputCol with HasYCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  val weightsLearningRate = new DoubleParam(
    this, "weightsLearningRate", "Learning rate for weights")

  def getWeightsLearningRate: Double = $(weightsLearningRate)

  def setWeightsLearningRate(value: Double): this.type = set(weightsLearningRate, value)

  setDefault(weightsLearningRate -> 1e-4)

  val biasLearningRate = new DoubleParam(
    this, "biasLearningRate", "Learning rate for bias")

  def getBiasLearningRate: Double = $(biasLearningRate)

  def setBiasLearningRate(value: Double): this.type = set(biasLearningRate, value)

  setDefault(biasLearningRate -> 1e-2)

  val stepsCnt = new IntParam(
    this, "stepsCnt", "Steps cnt")

  def getStepsCnt: Int = $(stepsCnt)

  def setStepsCnt(value: Int): this.type = set(stepsCnt, value)

  setDefault(stepsCnt -> 100)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getYCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val vectors: Dataset[(Vector, Vector)] = dataset.select(dataset($(inputCol)).as[Vector], dataset($(yCol)).as[Vector])

    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
      vectors.first()._1.size
    )

    // initialize weights and bias with ones
    var w: breeze.linalg.Vector[Double] = breeze.linalg.DenseVector.fill(dim) {
      1
    }
    var b = 1.0

    val N = dataset.count()
    val step_cnt = $(stepsCnt)

    for (_ <- 0 to step_cnt) {
      val (eps, epsdotx) =
        vectors.rdd
          .map {
            case (x, y) => (x.asBreeze, y.asBreeze)
          }
          .map({ case (x, y) =>
            val eps = (x dot w) + b - y(0)
            (eps, x * eps)
          }).reduce({ case ((eps1, epsdotx1), (eps2, epsdotx2)) => (eps1 + eps2, epsdotx1 + epsdotx2) })

      // 2 different learning rates as a workaround
      // In large datasets gradient for b is too small, if using the same value for both learning rates
      w = w - $(weightsLearningRate) / N * epsdotx
      b = b - $(biasLearningRate) / N * eps
    }

    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(w), b
    )).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val w: DenseVector,
                                           val b: Double) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(w: Vector, b: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), w.toDense, b)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(w, b), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf =
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => {
          (x.asBreeze dot w.asBreeze) + b
        })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = w.asInstanceOf[Vector] -> b

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  private implicit val vectorEncoder : Encoder[Vector] = ExpressionEncoder()
  private implicit val doubleEncoder : Encoder[Double] = ExpressionEncoder()

  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      val (w, b) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Double]).first()

      val model = new LinearRegressionModel(w, b)
      metadata.getAndSetParams(model)
      model
    }
  }
}
