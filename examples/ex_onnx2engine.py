from tensorrt_yolov8.engine_utils import engine_builder

if __name__ == "__main__":
    """
    Example of building an engine from an ONNX model 
    in FP16 precision and saving it to a file.
    """

    onnx_model_path = "yolov8s.onnx"

    # Build the engine
    engine_builder.build_engine_from_onnx(
        onnx_model_path, 
        save_engine_path="yolov8s_b1_fp16_export.engine",
        precision="fp16", 
        max_batch_size=1, 
    )