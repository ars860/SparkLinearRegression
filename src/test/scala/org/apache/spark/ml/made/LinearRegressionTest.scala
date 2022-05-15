package org.apache.spark.ml.made

import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val small_data: DataFrame = LinearRegressionTest._small_data
  lazy val vectors: Seq[(Vector, Vector)] = LinearRegressionTest._vectors

  "Model" should "predict z" in {
    // 1 * x + 2 * y + 3 * z + 4 = w
    val model: LinearRegressionModel = new LinearRegressionModel(
      w = Vectors.dense(1.0, 2.0, 3.0).toDense,
      b = 4.0
    ).setInputCol("x")
      .setYCol("y")
      .setOutputCol("z")

    validateModel(model, small_data)
  }

  "Estimator" should "learn weights and biases" in {
    val estimator = new LinearRegression()
      .setInputCol("x")
      .setYCol("y")
      .setOutputCol("z")
      .setWeightsLearningRate(1e-4)
      .setBiasLearningRate(2.5e-2)
      .setStepsCnt(100)

    val model = estimator.fit(data)

    model.w(0) should be(1.0 +- 1e-2)
    model.w(1) should be(2.0 +- 1e-2)
    model.b should be(3.0 +- 1e-1)
  }

  "Estimator" should "should produce functional model" in {
    val estimator = new LinearRegression()
      .setInputCol("x")
      .setYCol("y")
      .setOutputCol("z")
      .setWeightsLearningRate(1e-4)
      .setBiasLearningRate(2.5e-2)
      .setStepsCnt(100)

    val model = estimator.fit(data)

    validateModel(model, data)
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val processedData = model.transform(data)
    val vectors: Array[(Vector, Double)] = processedData.collect()
      .map(row => (row.getAs[Vector]("x"), row.getAs[Double]("z")))

    def f(x: Vector, w: Vector, b: Double) = (x.asBreeze dot w.asBreeze) + b

    for (vector <- vectors) {
      vector match {
        case (x, z) =>
          z should be(f(x, model.w, model.b))
      }
    }
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("x")
        .setYCol("y")
        .setOutputCol("z")
        .setWeightsLearningRate(1e-4)
        .setBiasLearningRate(2.5e-2)
        .setStepsCnt(100)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    model.w(0) should be(1.0 +- 1e-2)
    model.w(1) should be(2.0 +- 1e-2)
    model.b should be(3.0 +- 1e-1)

    validateModel(model, data)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("x")
        .setYCol("y")
        .setOutputCol("z")
        .setWeightsLearningRate(1e-4)
        .setBiasLearningRate(2.5e-2)
        .setStepsCnt(100)
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    val restoredModel: LinearRegressionModel = reRead.stages(0).asInstanceOf[LinearRegressionModel]

    restoredModel.w(0) should be(1.0 +- 1e-2)
    restoredModel.w(1) should be(2.0 +- 1e-2)
    restoredModel.b should be(3.0 +- 1e-1)

    validateModel(restoredModel, data)
  }
}

object LinearRegressionTest extends WithSpark {

  // function is x + 2 * y + 3 * z + 4 = w
  lazy val _small_vectors = Seq(
    (Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(18.0)),
    (Vectors.dense(11.0, 5.0, 1.0), Vectors.dense(28.0))
  )

  // 100 points for function x + 2 * y + 3 = z
  // generated with gen_data.py
  // should be stored in file, but I am lazy
  lazy val _vectors = Seq(
    (Vectors.dense(67.08692708902657, 16.49445889959288), Vectors.dense(103.07584488821233)),
    (Vectors.dense(69.84296792878301, 19.049229871736507), Vectors.dense(110.94142767225603)),
    (Vectors.dense(68.84061655118248, 35.260683076015695), Vectors.dense(142.36198270321387)),
    (Vectors.dense(67.50040324918474, 35.83177888756551), Vectors.dense(142.16396102431577)),
    (Vectors.dense(5.086231050557988, 51.540078731740024), Vectors.dense(111.16638851403803)),
    (Vectors.dense(27.235394751818966, 44.011976125841144), Vectors.dense(118.25934700350126)),
    (Vectors.dense(6.114216023258246, 73.98593105611958), Vectors.dense(157.08607813549742)),
    (Vectors.dense(62.938804061203925, 84.99286740472448), Vectors.dense(235.9245388706529)),
    (Vectors.dense(67.77373179885164, 97.1144270943528), Vectors.dense(265.00258598755727)),
    (Vectors.dense(80.70533529146627, 60.508645963767215), Vectors.dense(204.72262721900069)),
    (Vectors.dense(99.5792143513677, 25.875071699255802), Vectors.dense(154.3293577498793)),
    (Vectors.dense(63.77062463301332, 5.162097575124191), Vectors.dense(77.0948197832617)),
    (Vectors.dense(78.62176565459812, 90.25737879559654), Vectors.dense(262.1365232457912)),
    (Vectors.dense(57.37819344386891, 68.51299126040308), Vectors.dense(197.40417596467506)),
    (Vectors.dense(18.06046411816231, 48.69445124063346), Vectors.dense(118.44936659942924)),
    (Vectors.dense(42.35479495311457, 45.839435481551014), Vectors.dense(137.0336659162166)),
    (Vectors.dense(68.23196047287523, 94.87531277878784), Vectors.dense(260.98258603045093)),
    (Vectors.dense(29.911684946449746, 72.97017813713104), Vectors.dense(178.85204122071184)),
    (Vectors.dense(0.6588845802996679, 47.06075723598787), Vectors.dense(97.78039905227541)),
    (Vectors.dense(42.85897816834252, 67.8248133269439), Vectors.dense(181.50860482223032)),
    (Vectors.dense(27.11790586022268, 47.58477669705593), Vectors.dense(125.28745925433454)),
    (Vectors.dense(44.49364611340043, 15.2939171520314), Vectors.dense(78.08148041746323)),
    (Vectors.dense(35.55554821653198, 43.77404499276536), Vectors.dense(126.10363820206271)),
    (Vectors.dense(65.12810327413958, 44.138729104476646), Vectors.dense(156.40556148309287)),
    (Vectors.dense(38.457567385226575, 16.034023210816816), Vectors.dense(73.5256138068602)),
    (Vectors.dense(33.579774954208695, 30.25891276779419), Vectors.dense(97.09760048979707)),
    (Vectors.dense(12.646143417629851, 88.14190485232031), Vectors.dense(191.92995312227046)),
    (Vectors.dense(52.93557089557008, 23.469274795825278), Vectors.dense(102.87412048722064)),
    (Vectors.dense(58.74235755960952, 72.72986886986752), Vectors.dense(207.20209529934456)),
    (Vectors.dense(29.64045603206843, 51.36715464294003), Vectors.dense(135.3747653179485)),
    (Vectors.dense(53.20688382433855, 48.52187737078846), Vectors.dense(153.25063856591547)),
    (Vectors.dense(7.935155877398414, 79.92220590957363), Vectors.dense(170.77956769654568)),
    (Vectors.dense(69.66742650120912, 22.01667452369446), Vectors.dense(116.70077554859805)),
    (Vectors.dense(8.492451704233728, 79.87001855456319), Vectors.dense(171.2324888133601)),
    (Vectors.dense(65.8072335384481, 61.72965402812274), Vectors.dense(192.2665415946936)),
    (Vectors.dense(97.25951592951677, 21.181559113373815), Vectors.dense(142.62263415626438)),
    (Vectors.dense(39.65406483227489, 35.08957928421883), Vectors.dense(112.83322340071254)),
    (Vectors.dense(98.67222608370747, 28.356414325547895), Vectors.dense(158.38505473480325)),
    (Vectors.dense(63.05491635724437, 37.497731727223126), Vectors.dense(141.05037981169062)),
    (Vectors.dense(85.4200386851139, 31.141735394819303), Vectors.dense(150.7035094747525)),
    (Vectors.dense(10.707720428539302, 88.0612153862152), Vectors.dense(189.83015120096968)),
    (Vectors.dense(68.94130844277153, 6.786906490305511), Vectors.dense(85.51512142338255)),
    (Vectors.dense(27.82849659832547, 32.35338672646697), Vectors.dense(95.53527005125942)),
    (Vectors.dense(2.771744209040894, 79.87048614416578), Vectors.dense(165.51271649737245)),
    (Vectors.dense(86.34686232120507, 92.8377650860216), Vectors.dense(275.0223924932483)),
    (Vectors.dense(49.550546445392364, 73.51251034887437), Vectors.dense(199.57556714314111)),
    (Vectors.dense(29.156141936617907, 52.162579866496), Vectors.dense(136.4813016696099)),
    (Vectors.dense(69.25372325776428, 18.945016281169448), Vectors.dense(110.14375582010317)),
    (Vectors.dense(82.11081035849068, 55.48562754903603), Vectors.dense(196.08206545656276)),
    (Vectors.dense(64.5349985656901, 77.72483629986459), Vectors.dense(222.9846711654193)),
    (Vectors.dense(21.409666066065224, 13.76023881121473), Vectors.dense(51.93014368849468)),
    (Vectors.dense(74.30509429823886, 10.149678134904228), Vectors.dense(97.60445056804731)),
    (Vectors.dense(66.86457046145576, 27.967812259907387), Vectors.dense(125.80019498127052)),
    (Vectors.dense(38.364978100844695, 71.10194839187687), Vectors.dense(183.56887488459844)),
    (Vectors.dense(86.3739015852262, 43.87586047614007), Vectors.dense(177.12562253750633)),
    (Vectors.dense(53.992439849382244, 10.977704411430622), Vectors.dense(78.94784867224348)),
    (Vectors.dense(70.4515963489479, 14.736207575827498), Vectors.dense(102.9240115006029)),
    (Vectors.dense(66.90593613534976, 2.9134281111123483), Vectors.dense(75.73279235757445)),
    (Vectors.dense(50.61101863271542, 98.60324591454757), Vectors.dense(250.81751046181057)),
    (Vectors.dense(96.91035203125635, 85.08566472504634), Vectors.dense(270.08168148134905)),
    (Vectors.dense(58.51874174805592, 79.37188373509913), Vectors.dense(220.2625092182542)),
    (Vectors.dense(20.36721568123564, 90.65438826021148), Vectors.dense(204.6759922016586)),
    (Vectors.dense(43.41762411492422, 44.02448855614931), Vectors.dense(134.46660122722284)),
    (Vectors.dense(74.54396775035535, 29.05183635906313), Vectors.dense(135.6476404684816)),
    (Vectors.dense(90.27909150120797, 70.77241157631755), Vectors.dense(234.82391465384308)),
    (Vectors.dense(1.5842090477583226, 83.17042545309813), Vectors.dense(170.9250599539546)),
    (Vectors.dense(62.425684760745256, 86.24575353206177), Vectors.dense(237.91719182486878)),
    (Vectors.dense(17.77331644968524, 35.650341138699815), Vectors.dense(92.07399872708487)),
    (Vectors.dense(94.15264886468812, 14.996404907908467), Vectors.dense(127.14545868050504)),
    (Vectors.dense(35.25958369261916, 46.59547150992722), Vectors.dense(131.45052671247362)),
    (Vectors.dense(12.673535209229481, 67.4580185093526), Vectors.dense(150.58957222793467)),
    (Vectors.dense(95.42749154466755, 71.43577131671344), Vectors.dense(241.29903417809442)),
    (Vectors.dense(82.74274606779241, 99.76245063121077), Vectors.dense(285.26764733021395)),
    (Vectors.dense(89.68275070699568, 79.67986528862508), Vectors.dense(252.04248128424584)),
    (Vectors.dense(69.36078289049968, 92.54982068415319), Vectors.dense(257.46042425880603)),
    (Vectors.dense(55.79661918072449, 98.22501707337122), Vectors.dense(255.24665332746693)),
    (Vectors.dense(40.492784293279584, 10.151023678819627), Vectors.dense(63.79483165091884)),
    (Vectors.dense(5.676806028936376, 37.13981799280225), Vectors.dense(82.95644201454087)),
    (Vectors.dense(17.55412752801524, 95.92455950492858), Vectors.dense(212.4032465378724)),
    (Vectors.dense(74.30182256275195, 28.899589817572814), Vectors.dense(135.10100219789757)),
    (Vectors.dense(75.60424998940464, 49.475431318822096), Vectors.dense(177.55511262704883)),
    (Vectors.dense(18.428872880000448, 87.02270852649909), Vectors.dense(195.47428993299863)),
    (Vectors.dense(10.631536810865017, 5.33912552586806), Vectors.dense(24.309787862601137)),
    (Vectors.dense(72.02337794100927, 78.28027503247006), Vectors.dense(231.5839280059494)),
    (Vectors.dense(17.658697812686164, 20.886207423195348), Vectors.dense(62.43111265907686)),
    (Vectors.dense(73.27606173758356, 80.46051995634335), Vectors.dense(237.19710165027027)),
    (Vectors.dense(8.503546445155962, 16.693334596785427), Vectors.dense(44.890215638726815)),
    (Vectors.dense(68.59998968281789, 74.28314386630562), Vectors.dense(220.16627741542914)),
    (Vectors.dense(36.78079155663255, 77.55533865375631), Vectors.dense(194.89146886414517)),
    (Vectors.dense(24.691011709553244, 20.448828578470945), Vectors.dense(68.58866886649514)),
    (Vectors.dense(6.888432248151832, 60.213603537185435), Vectors.dense(130.3156393225227)),
    (Vectors.dense(43.65145152372475, 50.597744122932376), Vectors.dense(147.8469397695895)),
    (Vectors.dense(92.31480826079512, 37.85549586455254), Vectors.dense(171.02579998990018)),
    (Vectors.dense(23.76116836581478, 1.1134777890638103), Vectors.dense(28.988123943942398)),
    (Vectors.dense(47.870881654015186, 58.476193004755984), Vectors.dense(167.82326766352716)),
    (Vectors.dense(44.28892661854192, 79.70085581788557), Vectors.dense(206.69063825431306)),
    (Vectors.dense(83.60219235980789, 23.229503194038724), Vectors.dense(133.06119874788533)),
    (Vectors.dense(63.51872254662568, 12.831242399690112), Vectors.dense(92.1812073460059)),
    (Vectors.dense(85.866740131592, 56.38572149382812), Vectors.dense(201.63818311924825)),
    (Vectors.dense(70.14807837905252, 2.165470941165437), Vectors.dense(77.4790202613834))
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.toDF("x", "y")
  }

  lazy val _small_data: DataFrame = {
    import sqlc.implicits._
    _small_vectors.toDF("x", "y")
  }
}
