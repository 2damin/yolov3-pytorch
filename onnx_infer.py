import onnx
import onnxruntime
import argparse
import sys
import torch
import numpy as np

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
    
    x = torch.randn(1,3,608,608, requires_grad=True)
    
    print(onnx.checker.check_model(model))
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_session = onnxruntime.InferenceSession(args.model)

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    
    ort_outs = ort_session.run(None, ort_inputs)
    
    
    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    #np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    
    print("out : ", ort_outs)

if __name__ == "__main__":
    args = parse_args()
    main()