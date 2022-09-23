import pandas as pd
import numpy as np
import pyarrow as pa
from pyarrow import fs

# schema = pa.schema([
#     ("ints", pa.list_(pa.int32())),
#     ("ints", pa.list_(pa.int32()))
# ])

x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
y = np.array([[2, 2, 4, 5], [4, 5, 100, 6]], np.int32)
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
	for i in range(5):
		writer.write_batch(batch)

buf = sink.getvalue()
bhex = buf.hex()

with open("hex", "wb") as f:
	f.write(bhex)

s = buf.size

with pa.ipc.open_stream(buf) as reader:
	schema = reader.schema
	batches = [b for b in reader]

b = batches[0].to_pylist()
b

local = fs.LocalFileSystem()

with local.open_output_stream("test.arrow") as file:
	with pa.RecordBatchFileWriter(file, schema=batch.schema) as writer:
		writer.write_batch(batch)

with local.open_input_file("test.arrow") as file:
	with pa.RecordBatchFileReader(file) as reader:
		r = reader.read_all()
		rpd = r.to_pandas()
		rpd

# def nd_tolist():
#    r1 = np.random.random([, ])
