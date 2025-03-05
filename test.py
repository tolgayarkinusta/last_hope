from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
# Benchmark specific export format
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format="onnx")