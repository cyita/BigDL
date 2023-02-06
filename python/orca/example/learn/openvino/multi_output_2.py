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
from bigdl.orca.learn.openvino.new_est import Estimator
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