from bigdl.dllib.nn.layer import *
from pyspark.sql.functions import col, udf
from bigdl.orca.learn.openvino import Estimator
from pyspark.sql.types import ArrayType, FloatType
from bigdl.orca.data import SparkXShards

from bigdl.orca import init_orca_context, OrcaContext

sc = init_orca_context(cores=6, memory="5g")

spark1 = OrcaContext.get_spark_session()

batch_size = 5
model_path = "/d1/model/openvino2020_resnet50/resnet_v1_50.xml"

est = Estimator.from_openvino(model_path=model_path)

rdd = sc.range(0, 20, numSlices=5)
df = rdd.map(lambda x: [x, np.random.rand(150528).tolist()]).toDF(
	["index", "input"])  # .repartition(5)


def reshape(x):
	return np.array(x).reshape([3, 224, 224]).tolist()


reshape_udf = udf(reshape, ArrayType(ArrayType(ArrayType(FloatType()))))
df = df.withColumn("input", reshape_udf(df.input))
result_df = est.predict(df, batch_size=batch_size, feature_cols=["input"])
b = result_df.collect()
b


def to_shards(iter):
	cnt = 0
	index_list = []
	input_list = []
	for row in iter:
		index_list.append(row["index"])
		input_list.append(np.array(row["input"]))#.reshape([3, 224, 224]))
		cnt += 1
		if cnt == batch_size:
			yield {"index": index_list, "x": np.array(input_list)}
			cnt = 0
			index_list = []
			input_list = []
	if len(index_list) > 0:
		yield {"index": index_list, "x": np.array(input_list)}


shards = SparkXShards(df.rdd.mapPartitions(lambda iter: to_shards(iter)))
result_shards = est.predict(shards, batch_size=batch_size)
result_c = result_shards.collect()
result_c
