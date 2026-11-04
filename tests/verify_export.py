"""Checks the ONNX files with the official checker and onnxruntime, and the
generated Python scripts, against the JavaScript forward pass."""
import json, sys, importlib.util, numpy as np, onnx, onnxruntime as ort

d = sys.argv[1] if len(sys.argv) > 1 else "tests/.export"
ok = True
for name in ["ppo", "dqn", "ppo-cont", "evo"]:
    ref = json.load(open(f"{d}/{name}.json"))
    model = onnx.load(f"{d}/{name}.onnx")
    onnx.checker.check_model(model, full_check=True)
    sess = ort.InferenceSession(f"{d}/{name}.onnx")
    x = np.array(ref["inputs"], dtype=np.float32)
    res = sess.run(None, {"obs": x})
    err = np.abs(res[0] - np.array(ref["outputs"])).max()
    verr = np.abs(res[1][:, 0] - np.array(ref["values"])).max() if ref["hasCritic"] else 0.0
    spec = importlib.util.spec_from_file_location("brain", f"{d}/{name}_brain.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    perr = max(np.abs(mod.forward_numpy(xi) - np.array(o)).max() for xi, o in zip(x, ref["outputs"]))
    act = mod.choose_action(mod.forward_numpy(x[0]))
    good = err < 1e-4 and verr < 1e-4 and perr < 1e-4
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {name:9s} onnx max|Δ|={err:.2e} value|Δ|={verr:.2e} numpy|Δ|={perr:.2e} outputs={[o.name for o in model.graph.output]} action={act}")
sys.exit(0 if ok else 1)
