import base64

import pyarrow as pa
import numpy as np
from bigdl.dllib.utils.common import callBigDlFunc
from bigdl.orca import init_orca_context


def callJava(rdd, shapes):
	return callBigDlFunc("float", "arrowTest", rdd, ["a", "b"], shapes)


x = np.arange(24.0).reshape((2, 3, 4)).astype(np.float32)
y = np.arange(120.0).reshape((2, 3, 4, 5)).astype(np.float32)
shapelist = [list(x.shape), list(y.shape)]

# pa_tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
# print(pa_tensor)
data = pa.array([x.flatten(), y.flatten()])
batch = pa.record_batch([[x.flatten(), x.flatten() + 1],
                         [y.flatten(), y.flatten() + 1]], names=["a", "b"])
print(batch.num_rows)
print(batch.num_columns)
pat = pa.Table.from_batches(batches=[batch])
print(pat)
print(pat.to_pandas().to_string())
patpyl = pat.to_pylist()
a = data.tolist()
a

sink = pa.BufferOutputStream()

with pa.ipc.new_stream(sink, batch.schema) as writer:
	writer.write_batch(batch)

buf = sink.getvalue()
bhex = buf.hex()
encoding = 'utf-8'
bhex = bhex.decode(encoding)
# pyb = buf.to_pybytes()
# pybencode = base64.b64encode(pyb).decode("utf-8")

sc = init_orca_context(cores=4, memory="5g", conf={"spark.driver.maxResultSize": "5g"})
rdd = sc.parallelize([bhex, bhex], 2)
c = rdd.glom().collect()
c
r2 = callJava(rdd, shapelist)
r2.show()
