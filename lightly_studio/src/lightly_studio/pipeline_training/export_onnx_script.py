"""Exports a YOLOX checkpoint to ONNX. Run by the training interpreter, not by the backend.

The stock tools/export_onnx.py uses torch internals that no longer exist in torch >= 2.9,
hence this copy of the repository's export_onnx.py. The output has boxes decoded to pixels:
input "images" [batch, 3, H, W], output "output" [batch, anchors, 5 + classes].

    python export_onnx_script.py --exp-file exp.py --ckpt best_ckpt.pth --output model.onnx
"""

import argparse
import os
import sys

import onnxruntime as ort
import torch
from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

try:
    import onnx
    from onnxsim import simplify
except ImportError:  # The simplifier is optional.
    simplify = None


def main() -> int:
    """Loads the checkpoint, exports it and checks that onnxruntime loads the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    exp = get_exp(exp_file=args.exp_file)
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model = replace_module(model, torch.nn.SiLU, SiLU)
    model.eval()
    model.head.decode_in_inference = True
    print(
        f"checkpoint: {args.ckpt} (epoch {ckpt.get('start_epoch', '?')}), "
        f"classes: {exp.num_classes}"
    )

    dummy = torch.randn(1, 3, *exp.test_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    kwargs = {
        "input_names": ["images"],
        "output_names": ["output"],
        "dynamic_axes": {"images": {0: "batch"}, "output": {0: "batch"}},
        "opset_version": args.opset,
    }
    try:
        torch.onnx.export(model, dummy, args.output, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, args.output, **kwargs)
    print("exported:", args.output)

    if simplify is None:
        print("onnx-simplifier is not installed; kept the unsimplified graph")
    else:
        simplified, ok = simplify(onnx.load(args.output))
        if ok:
            onnx.save(simplified, args.output)
            print("simplified with onnx-simplifier")
        else:
            print("onnx-simplifier check failed; kept the unsimplified graph")

    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    print(
        "onnxruntime loads it:",
        [(i.name, i.shape) for i in session.get_inputs()],
        "->",
        [(o.name, o.shape) for o in session.get_outputs()],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
