"""Runs a curator in isolation: CPU only, no sockets. Called by run.py, not by hand."""
import importlib.util, json, os, resource, socket, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
def _no_net(*a, **k): raise RuntimeError("network disabled for curators")
socket.socket.connect = _no_net
socket.create_connection = _no_net
_cap = 16 << 30                     # address space cap in bytes: a curator that needs more gets a MemoryError, the machine is not run out of memory
resource.setrlimit(resource.RLIMIT_AS, (_cap, _cap))
curate_path, pool_dir, out = sys.argv[1:4]
sys.path.insert(0, os.path.dirname(os.path.abspath(curate_path)))
spec = importlib.util.spec_from_file_location("curate_user", curate_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
docs = json.load(open(f"{pool_dir}/docs.json"))
target = json.load(open(f"{pool_dir}/target_sample.json"))
idx = [int(i) for i in mod.select(docs, target)]
assert idx and all(0 <= i < len(docs) for i in idx), "select must return indices into docs"
json.dump(idx, open(out, "w"))
