/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.dllib.nn

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.dllib.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import java.security.SecureRandom

class AttentionSpec  extends FlatSpec with Matchers {


  val inputX : Tensor[Float] = Tensor(T(
    T(T( 2.43651805, -0.91763462, -0.79225763, -1.60945293, 1.29811144,
      -3.45230805, 2.61721765, -1.14181035),
      T( 0.47855864, -0.37405556, 2.19316191, -3.09021106, -0.48362581,
      -0.57608153, 1.70065416, -1.6498369),
      T(-0.25864231, -1.31678763, 0.06332062, 0.87422282, -1.65092877,
      1.71708556, 1.35238608, 0.75374151)),
      T(T( 1.35128392, -1.02559179, -0.18433534, -1.40365415, -0.40183212,
      0.7955332, -1.03749113, -0.59513029),
      T(-1.03075905, -1.26780846, -1.0068692, -0.0189969, -1.67596552,
      0.35162355, 2.48970327, 1.11306624),
      T(-0.28775333, -1.33144345, -1.12073744, 2.5386819, 0.07621163,
        -0.95549347, 0.28637323, 3.1503827))))

  val inputY : Tensor[Float] = inputX.clone()

  val inputBias : Tensor[Float] = Tensor(T(
      T(T(T( 0.06007948, 0.30860155, 0.15008516),
        T(-0.17612492, -0.5712591, -0.17467136),
        T(-0.10444712, 0.2933116, 0.41949171)),

      T(T( 0.46555104, 0.14279366, 0.44257058),
        T(-0.37719897, 0.62643408, 0.25646491),
        T(-0.14904642, 0.24425907, -0.03778586)),

      T(T( 0.56581469, 0.75990841, 1.0927877),
        T(-0.69824817, -0.7220569, -0.25223293),
        T( 0.08001853, 0.43808446, 0.15781747)),

      T(T(-1.01110061, -0.15310201, 0.41398732),
        T( 0.11504737, 0.38100559, -0.11116407),
        T(-0.10037903, 0.0932807, 0.20502582))),


      T(T(T( 0.09914986, 0.05950432, -0.33533114),
        T( 0.18878189, 0.06091064, 0.56474195),
        T( 0.59945894, 0.09257821, -0.18764248)),

        T(T(-0.3193652, 0.21174718, 0.03867003),
          T(-0.17192684, 0.02179843, -0.31000042),
          T( 0.34901602, -0.22356428, 0.61225385)),

        T(T( 0.20174582, 0.29678926, -0.54745592),
          T( 0.08469122, 0.37027823, -0.4768503),
          T(-0.13310925, 0.01630727, -0.68655866)),

        T(T( 0.1575797, 0.42308032, -0.42975797),
          T( 0.17527299, -0.65614171, -0.01934775),
        T(-0.80788618, 0.56070885, 0.20445027)))))

  val outputExpected : Tensor[Float] = Tensor[Float](
      T(T(T(-1.1452181, -1.4315172, -1.9381453, -0.8622163, -0.46799186,
        0.36559892, -0.20456696, 1.2694551),
        T(-0.2884395, -1.6821898, -1.9319419, -0.6244406, -2.2162085,
          1.1584786, -0.9849162, 1.7441965),
        T(-0.39340046, -1.2735752, -0.6773139, -0.6534181, -1.164992,
          0.16755454, -0.52454966, 0.10729646)),
        T(T(1.0821725, -0.10811941, -1.0524863, -1.0612597, -1.0610516,
          1.7031541, -0.31912667, -1.5727204),
        T(0.19246408, -0.4090905, -0.5103153, -0.71779424, -1.1626816,
          1.0925206, -0.4508822, 0.26590574),
        T(-0.48743886, -1.1587095, -2.0947423, -0.91988397, -1.5504212,
          0.78102344, -0.7083318, 0.8980509))))

  val weights: Table = T(
    "q" -> Tensor[Float](
        T(T(-0.372805, -0.57580054, -0.16542524, -0.29865405, 0.35401803, 0.15270126,
        -0.54465574, 0.15932709),
        T( 0.24691772, 0.30155098, 0.4186222, 0.2167002, 0.30048692, 0.27184665,
        0.39705545, -0.23575303),
        T( 0.00388521, 0.20807374, -0.378344, -0.30214158, -0.34708476, 0.04026955,
          -0.55643994, -0.5794907),
        T( 0.49563867, -0.20237926, -0.46280175, 0.28509408, 0.54167503, -0.3143271,
          -0.12728554, 0.38375044),
        T( 0.32280642, -0.5431511, 0.09327781, 0.26422644, -0.1516226, -0.592104,
          -0.4920348, -0.06154263),
        T(-0.3427992, -0.28234676, 0.60987645, -0.04226011, -0.4681016, -0.1708524,
        0.14569217, -0.08339447),
        T( 0.22560287, 0.35561, -0.50295657, 0.13627058, -0.3947954, 0.5856554,
          -0.4278072, -0.20018426),
        T(-0.262408, -0.21194538, -0.5646615, -0.50292665, -0.47206333, -0.5250014,
        0.26842934, 0.28272492))),
    "k" -> Tensor[Float](T(
        T(-0.343275, -0.5302577, 0.22225219, 0.22917205, -0.35248256, -0.52561647,
        -0.49496183, 0.19416988),
        T( 0.59556, 0.15709078, -0.5260543, 0.3003326, -0.4924144, 0.19770503,
        0.18886334, -0.4183287),
        T(-0.14076799, 0.20558482, -0.44356102, 0.3057044, -0.0961917, -0.41457063,
          -0.25426582, -0.43088654),
        T( 0.00211596, 0.5313905, 0.38138926, -0.53933024, 0.25935173, -0.4545771,
          -0.5513677, -0.42323098),
        T( 0.60221463, 0.46009654, -0.3742085, 0.30695522, -0.14824063, 0.08633447,
        0.5154777, -0.31166738),
        T( 0.5757794, -0.00155389, -0.27291873, 0.01211369, 0.10273433, -0.5679398,
          -0.4605189, -0.60379565),
        T(-0.2338264, -0.40447962, -0.20583275, 0.12039971, -0.4886889, -0.26302016,
        0.56051654, 0.0246914),
        T(-0.0083527, 0.07543635, 0.6011241, 0.5061092, -0.17393082, -0.02609855,
          -0.03866196, -0.47378802))),
    "v" -> Tensor[Float](
        T(T(-0.27888697, -0.3508993, 0.00061786, -0.05899942, -0.4096707, -0.59099805,
        0.00982529, 0.05359054),
        T( 0.3683961, -0.05546927, -0.2827503, 0.43347543, 0.1822511, -0.16377908,
          -0.5162845, -0.43161902),
        T( 0.46889406, 0.59701246, 0.48150903, 0.4334857, 0.486095, 0.53306824,
        0.27221018, 0.5941089),
        T( 0.12607813, -0.5313994, -0.57173353, -0.12448379, -0.11713088, -0.4439688,
          -0.527298, -0.37749383),
        T(-0.3919587, 0.05043119, 0.18434244, -0.01674193, -0.20570382, -0.21749035,
          -0.2891266, 0.12637317),
        T( 0.52648765, -0.07314765, 0.48385805, -0.03910315, 0.22911525, 0.01771665,
          -0.02246779, -0.40268806),
        T(-0.54250515, -0.31025118, -0.03211451, -0.12393585, -0.4777977, 0.18552327,
          -0.3151345, -0.5560428),
        T( 0.38067168, 0.45435983, 0.46077865, -0.10283256, -0.3396571, 0.26476836,
          -0.25029647, -0.5956288))),
    "output_transform" -> Tensor[Float](
        T(T(-0.22421107, 0.350811, 0.05354661, 0.6110292, -0.3909106, -0.5944199,
        0.10645795, 0.57871825),
        T(-0.5649649, -0.23917922, 0.3865885, 0.44473797, 0.29352474, -0.50426036,
          -0.3379699, 0.00927532),
        T(-0.37847292, -0.4825884, -0.05675334, -0.01127535, 0.08974767, -0.06169283,
        0.15506953, -0.02398986),
        T(-0.34070057, 0.12476408, 0.5375441, 0.2504276, 0.5667407, -0.599416,
        0.09187245, 0.5948525),
        T( 0.16609788, 0.55267304, 0.54386073, 0.18300432, 0.59399253, 0.02860391,
        0.26716715, -0.14422473),
        T( 0.41911787, -0.19523674, 0.4970067, 0.15865183, -0.46091762, 0.5183502,
          -0.2546733, 0.37238264),
        T(-0.23758182, 0.2648332, 0.14880872, -0.41371652, -0.52281517, 0.3087402,
          -0.4304452, -0.12153107),
        T( 0.02987367, 0.01645315, 0.58394355, 0.16796988, 0.23654258, -0.50470173,
        0.07536042, -0.5896087))))

  val gradWeightsExpected = T(
    "q" -> Tensor[Float](
        T(T(0.87246025, -3.906233, -2.200007, 2.4992926, -0.8980059,
          -1.802906, 3.1240094, 4.305592),
          T(1.3511171, -1.6406205, -0.4685239, -0.9951179, -0.5352191,
            -0.7400006, 2.1914737, -0.25137302),
          T(1.1200864, -0.83521426, 0.639663, -1.9919623, -0.45536512,
            0.5353197, -0.43531597, -0.9509695),
          T(-4.045067, 1.6682613, 0.78275317, 3.22646, -1.759268,
            5.1129704, -4.149416, 2.1751413),
          T(-9.566495, 6.2275248, 1.9285455, 6.6622696, -1.4173226,
            10.003353, -13.992972, 4.748796),
          T(4.1314225, -2.5120661, -0.9873823, -2.6948743, 0.7501687,
            -4.238539, 5.681432, -2.0739102),
          T(0.14631444, 3.20609, -1.6213762, 0.46593237, 5.3337274,
            -6.0198393, -4.384727, 0.9461689),
          T(0.9649465, 4.4181323, -0.5994254, -2.6823747, 5.786956,
            -5.894417, -5.1449404, -2.2193444))),
    "k" -> Tensor[Float](
        T(T(0.4311687, -0.39548448, 0.11438699, -0.52945966, -0.51122296,
          0.89019793, -0.15116894, -0.5108252),
        T(-1.1783814, -1.243825, -1.3546486, 1.6932042, -0.8066713,
          -0.66248465, 1.7828456, 2.703736),
        T(8.174547, -3.4214497, -3.6439848, -4.8295336, 4.798742,
          -13.629445, 10.977438, -3.1119957),
        T(-0.8001084, 0.7418808, 0.8360349, 0.17392862,
          -0.24244112, 1.1939042, -1.0853188, -0.19356316),
        T(0.5846084, 0.23388252, -0.40107524, 0.36903986,
          0.6782925, -0.19900057, -1.0856405, -0.09388584),
        T(12.200338, -2.9287677, -1.2595968, -12.979106,
          5.6081734, -15.572923, 13.633935, -10.663923),
        T(-0.7373574, 0.40913522, -0.23541816, 0.036719695,
          0.04470883, -0.29267675, 0.06379409, 0.28098464),
        T(-3.5493982, -0.31215632, -1.1058445, 0.8615737,
          -2.4665074, 0.884194, 3.4850554, 1.9124848))),
    "v" -> Tensor[Float](
        T(T(-0.4765153, 6.1380205, 3.2578008, 0.9474751, 4.59338,
          -1.7985046, -3.8672557, -3.5786848),
        T(-0.7310653, 7.6148925, 4.259535, 0.19133823, 5.01503,
          -1.5798602, -4.0146008, -5.356738),
        T(2.0231, -0.31761393, 3.31298, -6.174352, -0.09106046,
          -1.867869, 3.410323, -4.3797026),
        T(-0.85167503, 8.095302, 2.0130663, -3.0256634, 2.127143,
          4.2343884, -6.077573, -9.252485),
        T(-17.551378, 13.798134, 0.17861271, 18.369804, -4.2673993,
          26.651054, -27.69285, 5.5386744),
        T(4.834157, -3.8755064, 1.0489246, -7.2673984, 0.4043241,
          -6.9056253, 8.62311, -2.964557),
        T(0.12066603, -5.2807674, -1.4196055, -0.22159815,
          -5.092498, 3.3522754, 4.918932, 2.1831658),
        T(1.3403978, 13.126951, 2.5753145, -2.8867948, 14.16077,
          -11.119917, -13.125348, -6.7138343))),
    "output_transform" -> Tensor[Float](
        T(T(-0.0042502284, -0.11399703, -2.238832, -3.1139684, 4.217409,
          0.5355667, -0.0063366815, 3.1841836),
        T(1.3698364, 0.80534774, -3.1648421, -1.3512311, 11.648689,
          -0.83019644, -1.2784916, 6.9636717),
        T(2.2579505, 1.050436, -3.0892425, 0.18534195, 14.036037,
          -0.430179, -1.587869, 9.792297),
        T(1.5637071, 0.9522813, -1.3883977, 1.2237415, 6.60323,
          -0.36405873, -1.2773074, 3.975604),
        T(1.7391877, 1.2802999, -1.8425734, 1.3109185, 12.5123205,
          -2.1025999, -1.9961238, 6.978856),
        T(-1.6705356, -1.0838037, 0.3917895, -2.9463623, -6.181151,
          1.1864538, 1.4216664, -4.0609703),
        T(0.667181, 0.5003866, -0.87703246, 0.28493565, 5.563534,
          -0.8977524, -0.8192562, 3.1073413),
        T(0.15117985, 0.107112974, 2.9966583, 4.1665463, -8.682102,
          0.6451258, 0.2614205, -6.8862433)))
  )
  "attention layer" should "work correctly" in {
    // compare with tensorflow 1.13.1
    val attention = new Attention[Float](8, 4, 1.0f)

    val paramsTable = attention.getParametersTable()
    val w1 = weights.get[Tensor[Float]]("q").get
    val w2 = weights.get[Tensor[Float]]("k").get
    val w3 = weights.get[Tensor[Float]]("v").get
    val w4 = weights.get[Tensor[Float]]("output_transform").get
    for (i <- paramsTable.keySet) {
      val params = paramsTable.get[Table](i).get.get[Tensor[Float]]("weight").get
      if (i.toString contains "_q") {
        params.copy(w1.t())
      } else if (i.toString contains "_k") {
        params.copy(w2.t())
      } else if (i.toString contains "_v") {
        params.copy(w3.t())
      } else if (i.toString contains "_output_transform") {
        params.copy(w4.t())
      }
    }

    val output = attention.forward(T(inputX, inputY, inputBias))
    val gradInput = attention.backward(T(inputX, inputY, inputBias), output)

    output should  be(outputExpected)
    // gradInput should be(gradInputExpected)

    val gw1 = gradWeightsExpected.get[Tensor[Float]]("q").get
    val gw2 = gradWeightsExpected.get[Tensor[Float]]("k").get
    val gw3 = gradWeightsExpected.get[Tensor[Float]]("v").get
    val gw4 = gradWeightsExpected.get[Tensor[Float]]("output_transform").get
    for (i <- paramsTable.keySet) {
      val params = paramsTable.get[Table](i).get.get[Tensor[Float]]("gradWeight").get
      if (i.toString contains "_q") params should be(gw1)
      if (i.toString contains "_k") params should be(gw2)
      if (i.toString contains "_v") params should be(gw3)
      if (i.toString contains "_output_transform") params should be(gw4)
    }
  }
}

class AttentionSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val attention = new Attention[Float](8, 4, 1.0f).setName("attention")
    val inputX = Tensor[Float](2, 3, 8).apply1(_ => new SecureRandom().nextFloat())
    val inputY = inputX.clone()
    val inputBias = Tensor[Float](2, 4, 3, 3).apply1(_ => new SecureRandom().nextFloat())
    runSerializationTest(attention, T(inputX, inputY, inputBias))
  }
}