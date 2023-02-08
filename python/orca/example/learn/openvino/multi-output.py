import os
import tarfile
from bigdl.dllib.feature.dataset.base import maybe_download
import tempfile
import shutil
import time
import numpy as np
from pyspark.sql import SparkSession
from bigdl.orca.learn.utils import dataframe_to_xshards

from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.data import SparkXShards
from bigdl.orca.learn.openvino import Estimator
from bigdl.orca import init_orca_context

data_url = "https://sourceforge.net/projects/analytics-zoo/files/"
local_path = "/tmp/tmpyhny717gopenvino"

def load_multi_output_model():
    os.makedirs(local_path, exist_ok=True)
    model_url = data_url + "/analytics-zoo-data/ov_multi_output.tar"
    model_path = maybe_download("ov_multi_output.tar",
                                local_path, model_url)
    tar = tarfile.open(model_path)
    tar.extractall(path=local_path)
    tar.close()
    model_path = os.path.join(local_path, "FP32/model_float32.xml")
    return model_path

conf = {"spark.driver.maxResultSize": "3g"}
sc = init_orca_context(cores=4, spark_log_level="WARN", memory="10g", conf=conf)

model_path = load_multi_output_model()
spark = SparkSession(sc)
est = Estimator.from_openvino(model_path=model_path)

data = np.random.rand(3, 550, 550)
rdd = sc.range(0, 2, numSlices=1)
df = rdd.map(lambda x: [data.tolist()]).toDF(["input"])

result_df = est.predict(df, feature_cols=["input"])
result_df2 = est.predict(df, feature_cols=["input"], original=True)
result_df2.show()

# df_c = result_df.rdd.map(lambda row: [row[1], row[2], row[3], row[4]]).collect()
# df_c = [np.concatenate((np.array(df_c[0][i]), np.array(df_c[1][i]))) for i in range(4)]

# df_c2 = result_df2.rdd.map(lambda row: [row[1], row[2], row[3], row[4]]).collect()
# df_c2 = [np.concatenate((np.array(df_c2[0][i]), np.array(df_c2[1][i]))) for i in range(4)]
# assert np.all([np.allclose(r1, r2) for r1, r2 in zip(df_c, df_c2)])

# shards, _ = dataframe_to_xshards(df,
#                                     validation_data=None,
#                                     feature_cols=["input"],
#                                     label_cols=None,
#                                     mode="predict")
# result_shard = est.predict(shards, batch_size=4)
# shard_c = result_shard.collect()[0]
# nd_input = np.squeeze(np.array(df.select('input').collect()))

# nd_input = np.random.rand(200, 3, 550, 550)
# t1 = time.perf_counter()
# result_np = est.predict(nd_input)
# t2 = time.perf_counter()
# print(f"E2E time: {t2 - t1}s")
# result_np

# assert np.all([np.allclose(r1, r2) for r1, r2 in zip(df_c, result_np)])
# assert np.all([np.allclose(r1, r2) for r1, r2 in zip(shard_c, result_np)])