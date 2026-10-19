# AR5 launch recipe (needs one permission rule from Rustom)

The auto-mode classifier blocks the Workflow tool for this script and blocks edits to the settings files from the session, so the rule must be added by hand.

## 1. Permission rule (one line, in an interactive Claude Code session or by editing the file)

Add "Workflow" to the allow list in /homes/rdubash/.claude/settings.local.json (project-local settings for the home-directory project), for example:

    {"permissions": {"allow": ["Workflow"]}}

If the file already has a permissions block, append "Workflow" to its allow array. Alternative: in an interactive session run /permissions and add the tool Workflow to the allow list. Remove it again after the run.

## 2. Pilot (one iteration, cherry, throwaway tree, about 20 minutes, under 1 USD)

Workflow tool, scriptPath /homes/rdubash/active-audit/avrl/autoresearch.live.js, args:

    {"runs":[{"run_id":"P0TEST","base":"/homes/rdubash/avrl_live/P0TEST","models":{"plan":"haiku","run":"haiku","analyze":"haiku","critic":"sonnet","screen":"haiku"},"critic_mode":"cascade"}],"start_epoch":<date +%s at launch>,"stop_after_s":14400,"max_iters":1}

Pass criteria: the planner picks a backlog experiment; the runner launches through BASE/bin/gpu_run.sh and sees done.flag; the analyst appends one entry to P0TEST/avrl/RESEARCH_LOG.md; the two screeners return; the leak scan (exp/ar5_leak_scan.py on the workflow transcript dir) shows no hard hit outside the tool-results spill; exp/ar5_cost_from_transcripts.py parses the labels.

## 3. The block (five arms, 8 hours)

Before launch: run exp/ar5_make_run.sh for FULL cherry, ROUTED willow, CHEAP walnut, CASCADE pine, FULL_R spruce (fresh trees from the patched seed); confirm each host shows under 2 GB GPU memory used and no compute apps; start the snapshotter: setsid nohup bash exp/ar5_snapshot.sh /homes/rdubash/avrl_live/{FULL,ROUTED,CHEAP,CASCADE,FULL_R} and write T0 into each tree (the driver script does this when run; for the Workflow route write BASE/T0 by hand with the start epoch). Fill the launch facts in exp/AR5_PREREG.md (seed manifest sha256, script sha256, key sha256, T0, resolved model ids after the first transcripts) and append the pre-registration entry to avrl/RESEARCH_LOG.md before the first agent starts.

Workflow tool, same scriptPath, args:

    {"runs":[
      {"run_id":"FULL","base":"/homes/rdubash/avrl_live/FULL","models":{"plan":"sonnet","run":"sonnet","analyze":"sonnet","critic":"sonnet","screen":"haiku"},"critic_mode":"single"},
      {"run_id":"ROUTED","base":"/homes/rdubash/avrl_live/ROUTED","models":{"plan":"haiku","run":"haiku","analyze":"haiku","critic":"sonnet","screen":"haiku"},"critic_mode":"single"},
      {"run_id":"CHEAP","base":"/homes/rdubash/avrl_live/CHEAP","models":{"plan":"haiku","run":"haiku","analyze":"haiku","critic":"haiku","screen":"haiku"},"critic_mode":"single"},
      {"run_id":"CASCADE","base":"/homes/rdubash/avrl_live/CASCADE","models":{"plan":"haiku","run":"haiku","analyze":"haiku","critic":"sonnet","screen":"haiku"},"critic_mode":"cascade"},
      {"run_id":"FULL_R","base":"/homes/rdubash/avrl_live/FULL_R","models":{"plan":"sonnet","run":"sonnet","analyze":"sonnet","critic":"sonnet","screen":"haiku"},"critic_mode":"single"}],
     "start_epoch":<date +%s at launch>,"stop_after_s":26100,"max_iters":40}

Stop rules: the gate stops new iterations at T0 + 7 h 15 min; hard-stop the workflow (TaskStop) at T0 + 8 h 30 min if still running; touching BASE/STOP stops one arm at its next iteration. Primary analysis horizon T0 + 8 h from the snapshots.

## 4. After the run

exp/ar5_leak_scan.py <transcript dir>; exp/ar5_cost_from_transcripts.py <transcript dir> --models <per-run models JSON>; exp/ar5_gate.py and exp/ar5_adjudicate.py per run and ledger prefix; exp/ar5_trace.py and exp/ar5_audit.py on every entry; originals integrity check (sha256 of avrl/RESEARCH_LOG.md, lit/avrl/EXPERIMENT_BACKLOG.md, avrl/PROJECT.md unchanged); copy run trees to avrl/runs/AR5/<run_id>/; write the AR5 result entry.
