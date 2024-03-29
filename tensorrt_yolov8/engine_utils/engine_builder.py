import tensorrt as trt


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