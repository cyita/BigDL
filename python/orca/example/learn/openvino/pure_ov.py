import time
from openvino.inference_engine import IECore
from memory_profiler import profile
import numpy as np

ie = IECore()
config = {'CPU_THREADS_NUM': "8"}
ie.set_config(config, 'CPU')
net = ie.read_network(
	model="/home/yina/Documents/data/myissue/openvino_model/FP32/model_float32.xml",
	weights="/home/yina/Documents/data/myissue/openvino_model/FP32/model_float32.bin")
net.batch_size = 16
local_model = ie.load_network(network=net, device_name="CPU", num_requests=1)
infer_request = local_model.requests[0]
inputs = list(net.input_info.keys())
output_dict = {k: v.shape for k, v in net.outputs.items()}
outputs = list(output_dict.keys())


@profile
def inference():
	# input = {inputs[0]: np.random.rand(40, 3, 550, 550)}
	input = np.random.rand(64, 3, 550, 550)
	arrays = np.array_split(input, 4)
	for i in arrays:
		i = {inputs[0]: i}
		t1 = time.time()
		infer_request.infer(i)
		pred = list(map(lambda output:
		                infer_request.output_blobs[output].buffer,
		                outputs))
		t2 = time.time()
		print("--------------inference: ", t2 - t1)
	return pred


for i in range(5):
	result = inference()
	for r in result:
		print(r.shape)
	print("------------------------")

