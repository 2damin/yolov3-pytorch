import onnx
import onnxruntime
import argparse
import sys
import numpy as np
import tensorrt as trt

def parse_args():
    parser = argparse.ArgumentParser(description="onnx_inference")
    parser.add_argument("--gpus", type=int, nargs='+', default=[], help="List of device ids.")
    parser.add_argument('--model', type=str, help="model path",
                        default=None, dest='model')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def main():
    print("main")
    
    model = onnx.load(args.model)
    
    x = np.randn([1,3,608,608])
    
    print(onnx.checker.check_model(model))
    

    ort_session = onnxruntime.InferenceSession(args.model)

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    
    ort_outs = ort_session.run(None, ort_inputs)

    print("out : ", ort_outs)

if __name__ == "__main__":
    args = parse_args()
    main()