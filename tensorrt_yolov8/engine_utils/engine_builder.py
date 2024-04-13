import tensorrt as trt

def build_engine_from_onnx2(
		onnx_path:str,
		save_engine_path:str,
		shape=(1, 3, 640, 640),
		dynamic:bool=False,
		half:bool=False,
		verbose:bool=False,
):
	
	# TODO: check input parameters
 
	logger = trt.Logger(trt.Logger.INFO)
	
	if verbose:
		logger.min_severity = trt.Logger.Severity.VERBOSE

	builder = trt.Builder(logger)
	config = builder.create_builder_config()

	workspace_size_gb = 1
	config.max_workspace_size = int(workspace_size_gb * (1 << 30))

	flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
	network = builder.create_network(flag)
	parser = trt.OnnxParser(network, logger)
	if not parser.parse_from_file(onnx_path):
		raise RuntimeError("Failed to parse ONNX file")
	
	inputs = [network.get_input(i) for i in range(network.num_inputs)]
	outputs = [network.get_output(i) for i in range(network.num_outputs)]
	for inp in inputs:
		print(f"[INPUTS] {inp.name} with shape {inp.shape} of type {inp.dtype}")
	for out in outputs:
		print(f"[OUTPUTS] {out.name} with shape {out.shape} of type {out.dtype}")

	if dynamic:
		if shape[0] <= 1:
			print("Dynamic=True model requires max batch size, i.e. 'batch=16'")
		
		profile = builder.create_optimization_profile()
		for inp in inputs:
			profile.set_shape(inp.name, min=(1, *shape[1:]), opt=(max(1, shape[0] // 2), *shape[1:]), max=shape)
		config.add_optimization_profile(profile)
	
	half = True
	if builder.platform_has_fast_fp16 and half:
		config.set_flag(trt.BuilderFlag.FP16)

	# int8 = False
	# if builder.platform_has_fast_int8 and int8:
	# 	config.set_flag(trt.BuilderFlag.INT8)
	# 	# missing calibrator!! see how
	# 	config.int8_calibrator = 
 
	with builder.build_engine(network, config) as engine, open(save_engine_path, "wb") as f:
		f.write(engine.serialize())
		print(f"Engine built and saved to {save_engine_path}")



# TODO: different batch sizes for min, opt, max
# still result in engine with batch size 1...
# using trtexec with --minShapes=input_layer_name:4x3x640x640
# works as expected and sets the batch size to 4
def build_engine_from_onnx(
		onnx_path:str,
		save_engine_path:str,
		min_batch_size: int=1,
		opt_batch_size: int=1,
		max_batch_size: int=1,
		precision:str = "fp32",
		logger_level:str = "INFO", # "WARNING", "INFO", "VERBOSE", "ERROR",
		**kwargs,
	) -> None:
	"""
	Builds a TensorRT engine from an ONNX model and saves it to a file.

	Args:
	- onnx_path (str): Path to the ONNX model file.
	- save_engine_path (str): Path to save the built engine.
	- min_batch_size (int, optional): Minimum batch size. Defaults to 1.
	- opt_batch_size (int, optional): Optimal batch size. Defaults to 1.
	- max_batch_size (int, optional): Maximum batch size. Defaults to 1.
	- precision (str, optional): Precision of the engine. Must be one of 'fp32', 'fp16', 'int8'. Defaults to "fp32".
	- logger_level (str, optional): Logger level. Must be one of 'WARNING', 'INFO', 'VERBOSE', 'ERROR'. Defaults to "INFO".
	- **kwargs: Additional keyword arguments.
	"""
	
	if precision not in ["fp32", "fp16", "int8"]:
		raise ValueError(f"Invalid precision: {precision} - must be one of 'fp32', 'fp16', 'int8'")

	if logger_level == "WARNING":
		logger_type = trt.Logger.WARNING
	elif logger_level == "INFO":
		logger_type = trt.Logger.INFO
	elif logger_level == "VERBOSE":
		logger_type = trt.Logger.VERBOSE
	elif logger_level == "ERROR":
		logger_type = trt.Logger.ERROR
	else:
		raise ValueError(f"Invalid logger level: {logger_level} - must be one of 'WARNING', 'INFO', 'VERBOSE', 'ERROR'")
	
	EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

	TRT_LOGGER = trt.Logger(logger_type)
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
		config = builder.create_builder_config()
		# config.max_workspace_size = 1 << 30 # Deprecated
		# config.set_memory_pool_limit(tensorrt.Memory1 << 30)

		profile = builder.create_optimization_profile()
		profile.set_shape("images", [min_batch_size, 3, 640, 640], [opt_batch_size, 3, 640, 640], [max_batch_size, 3, 640, 640])
		config.add_optimization_profile(profile)

		if precision == "fp16":
			config.flags = 1 << int(trt.BuilderFlag.FP16)
		elif precision == "int8":
			# config.flags = 1 << trt.BuilderFlag.INT8
			raise NotImplementedError("Int8 precision not yet supported.")

		# Parse the ONNX file
		with open(onnx_path, 'rb') as model:
			parser.parse(model.read())

		# Build and return an engine
		engine = builder.build_serialized_network(network, config) 
		
		with open(save_engine_path, "wb") as f:
			f.write(engine)

		print(f"Engine built and saved to {save_engine_path}")

if __name__ == "__main__":

	onnx_path = "/home/Documents/Experiments/TENSORRT/tensorrt_yolov8/examples/yolov8s.onnx"
	engine_path = "/home/Documents/Experiments/TENSORRT/tensorrt_yolov8/examples/m_export_yolov8s_fp16_dynamic.onnx"

	build_engine_from_onnx2(
		onnx_path=onnx_path,
		save_engine_path=engine_path,
		shape=(4, 3, 640, 640),
		dynamic=True,
		half=False,
	)