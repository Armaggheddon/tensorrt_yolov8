from tensorrt_yolov8.engine_utils import engine_builder
from tensorrt_yolov8 import Pipeline

if __name__ == "__main__":
    """
    Example of building an engine from an ONNX model 
    in FP16 precision and saving it to a file.
    """

    # onnx_model_path = "/home/Documents/Experiments/TENSORRT/tensorrt_yolov8/examples/yolov8s.onnx"

    # Parameter --minShapes=input_layer_name:4x3x640x640 sets the batch size to 4

    # # Build the engine
    # engine_builder.build_engine_from_onnx(
    #     onnx_model_path, 
    #     save_engine_path="yolov8s_b1_fp16_export.engine",
    #     precision="fp16", 
    #     min_batch_size=2,
    #     opt_batch_size=2,
    #     max_batch_size=2, 
    # )

    # Check that shape are correct
    det_pipe = Pipeline("detection", "/home/Documents/Experiments/TENSORRT/tensorrt_yolov8/examples/v8s_b4_fp32.engine")
    print(det_pipe._Pipeline__engine.input_shapes)
    print(det_pipe._Pipeline__engine.output_shapes)

