#!/usr/bin/env python3
"""Step 6 loops: the routed loop of C1 and R1 (routed.py, unchanged) with two additions.

    python curation/loop/routed_s6.py [--memory FILE] <the arguments of routed.py>

1. The model list is models_s6.json, which adds the Qwen 4B planner served from cherry, plain and with the trained
   adapter.
2. --memory FILE puts the text of FILE (ideas kept in earlier runs) in front of every planner prompt of the loop.
   The text is copied to memory.txt in the run directory, so a resumed loop uses the same text.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine as E     # noqa: E402
import routed          # noqa: E402

E.call_mod.REGISTRY_PATH = HERE / "models_s6.json"
MEMORY = None


class MemoryLoop(routed.RoutedLoop):
    def __init__(self, cfg, models=None, runner=None):
        super().__init__(cfg, models, runner)
        kept = self.dir / "memory.txt"
        if MEMORY and not kept.exists():
            kept.write_text(Path(MEMORY).read_text())
        self.memory = kept.read_text() if kept.exists() else None

    def plan_prompt(self):
        prompt = super().plan_prompt()
        return self.memory.rstrip() + "\n\n" + prompt if self.memory else prompt


def main(argv=None):
    global MEMORY
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--memory" in argv:
        i = argv.index("--memory")
        MEMORY = argv[i + 1]
        del argv[i:i + 2]
    routed.RoutedLoop = MemoryLoop
    return routed.main(argv)


if __name__ == "__main__":
    sys.exit(main())
