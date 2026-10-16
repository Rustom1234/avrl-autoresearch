export const meta = {
  name: 'avrl-autoresearch',
  description: 'Autonomous research loop: pick the next pre-registered experiment, run it, analyze, adversarially verify, log to the ledger, propose follow-ups, repeat until dry',
  phases: [
    { title: 'Plan', detail: 'choose the next experiment from the backlog and ledger' },
    { title: 'Run', detail: 'implement and run it on goldberg (or hannah if marked A100)' },
    { title: 'Analyze', detail: 'write the ledger entry with verdict and follow-ups' },
    { title: 'Verify', detail: 'critic checks numbers, verdict and confounds' },
  ],
}

const ROOT = '/homes/rdubash/active-audit/avrl'
const LEDGER = `${ROOT}/RESEARCH_LOG.md`
const BACKLOG = '/homes/rdubash/active-audit/lit/avrl/EXPERIMENT_BACKLOG.md'
const SYNTH = '/homes/rdubash/active-audit/lit/avrl/SYNTHESIS.md'
const MAX_ITERS = (args && args.max_iters) || 8

const RULES = `Ground rules. Karpathy: think before coding, simplest code that works, surgical changes, concrete success criteria. Never use em dashes anywhere. Write only under ${ROOT}/ (new scripts go in ${ROOT}/exp/, results in ${ROOT}/runs/<exp_id>/). Never git push. Never touch anything outside active-audit. Compute rules on goldberg (this machine, one RTX 5090 32GB): one GPU job at a time, check 'nvidia-smi --query-gpu=memory.used --format=csv,noheader' and 'pgrep -af python' before launching and wait if something is using more than 2 GB; export CPATH=$HOME/.local/include/python3.12 before any GPU python (Triton needs it); use /homes/rdubash/active-audit/.venv-judge/bin/python for vLLM generation scripts and /homes/rdubash/active-audit/.venv/bin/python for analysis (sentence-transformers, sklearn, pandas, transformers); launch long jobs detached with 'setsid nohup bash -c "..." > log 2>&1 < /dev/null & disown' and a done.flag, then poll with 'until [ -f <flag> ]; do sleep 30; done' inside Bash calls of at most 540 seconds, repeating the call as needed; hard cap 45 minutes per experiment, kill the job if exceeded (pkill -f '<script name>[.]py' with the bracket trick so you do not kill your own shell). hannah (4x A100 40GB) only for experiments marked A100 and only via ssh -o BatchMode=yes rustom@hannah and srun -p gpu (no --gres); the phase-1 GRPO harness may or may not exist at ~/avrl/p1_grpo.py on hannah, check before planning around it. Existing assets: ${ROOT}/p0_rollouts.py (vLLM rollouts, majority-vote proxy vs gold), ${ROOT}/p0_analyze.py (structure metrics and allocation replay via /homes/rdubash/active-audit/runs/sim_generic.py), models under /homes/rdubash/active-audit/models/, data under ${ROOT}/data/, results under ${ROOT}/runs/. Copy and modify scripts rather than editing p0 scripts in place. Report failures honestly with the exact error text.`

const SCHEMA_PLAN = { type: 'object', properties: { id: { type: 'string' }, hypothesis: { type: 'string' }, plan: { type: 'string' }, command_shape: { type: 'string' }, metric: { type: 'string' }, pass_threshold: { type: 'string' }, kill_threshold: { type: 'string' }, expected_minutes: { type: 'number' }, machine: { type: 'string', enum: ['goldberg', 'hannah', 'none'] }, why_now: { type: 'string' } }, required: ['id', 'hypothesis', 'plan', 'command_shape', 'metric', 'pass_threshold', 'kill_threshold', 'expected_minutes', 'machine', 'why_now'] }
const SCHEMA_RUN = { type: 'object', properties: { ok: { type: 'boolean' }, result_files: { type: 'array', items: { type: 'string' } }, scripts_written: { type: 'array', items: { type: 'string' } }, key_numbers: { type: 'string' }, wall_minutes: { type: 'number' }, error: { type: 'string' } }, required: ['ok', 'result_files', 'scripts_written', 'key_numbers', 'wall_minutes', 'error'] }
const SCHEMA_ANALYSIS = { type: 'object', properties: { verdict: { type: 'string', enum: ['pass', 'kill', 'inconclusive', 'failed'] }, one_line: { type: 'string' }, follow_ups: { type: 'array', items: { type: 'string' } }, thesis_impact: { type: 'string' } }, required: ['verdict', 'one_line', 'follow_ups', 'thesis_impact'] }
const SCHEMA_CRITIC = { type: 'object', properties: { ok: { type: 'boolean' }, issues: { type: 'array', items: { type: 'string' } }, corrected_verdict: { type: 'string' } }, required: ['ok', 'issues', 'corrected_verdict'] }

const done = []
let dry = 0
for (let i = 0; i < MAX_ITERS && dry < 2; i++) {
  phase('Plan')
  const plan = await agent(`${RULES}\n\nIteration ${i + 1} of at most ${MAX_ITERS}. Read ${BACKLOG}, ${SYNTH} (skim), ${ROOT}/PROJECT.md, ${LEDGER} if it exists (it lists what has been run and the follow-ups proposed), and the newest result json files under ${ROOT}/runs/ (ls -t). Pick the single next experiment with the highest information value per GPU hour for the thesis claims that is runnable NOW: on goldberg in under 45 minutes, or on hannah if the backlog marks it A100 and the harness exists there. Do not repeat an experiment already in the ledger unless the ledger asks for a replication with more seeds. Prefer experiments that can kill a claim over ones that can only confirm it. If nothing runnable remains, return id NONE and say why. Already run this loop: ${JSON.stringify(done)}.`, { label: `plan:${i + 1}`, phase: 'Plan', schema: SCHEMA_PLAN, model: 'sonnet' })
  if (!plan || plan.id === 'NONE' || plan.machine === 'none') { dry++; log(`iteration ${i + 1}: nothing runnable (${plan ? plan.why_now : 'planner died'})`); continue }
  dry = 0
  log(`iteration ${i + 1}: running ${plan.id} on ${plan.machine}: ${plan.hypothesis}`)

  phase('Run')
  const run = await agent(`${RULES}\n\nImplement and run experiment ${plan.id}. Spec: ${JSON.stringify(plan)}. Write any new script to ${ROOT}/exp/${plan.id}.py (or .sh), results to ${ROOT}/runs/${plan.id}/ including a result.json with the metric values and a done.flag when finished. Reuse the p0 scripts by copying; keep changes minimal. Before launching, check the GPU is free. Launch detached, poll until done, cap at 45 minutes. If it crashes, read the traceback, fix once, rerun once; if it fails again return ok=false with the exact error. When done, return the result file paths and a compact excerpt of the key numbers (the metric, the baselines, sample sizes).`, { label: `run:${plan.id}`, phase: 'Run', schema: SCHEMA_RUN, model: 'sonnet' })
  if (!run || !run.ok) { log(`${plan.id} failed: ${run ? run.error : 'runner died'}`) }

  phase('Analyze')
  const analysis = await agent(`${RULES}\n\nExperiment ${plan.id} finished. Spec: ${JSON.stringify(plan)}. Run report: ${JSON.stringify(run)}. Read the result files and logs under ${ROOT}/runs/${plan.id}/ yourself, do not trust the run report's numbers without checking. Append an entry to ${LEDGER} (create the file with a short header if it does not exist) in this exact shape: '## <id> <date from the bash date command> <verdict>' then hypothesis, what was run (model, task, n, seeds, budget), a markdown table of the numbers with baselines, verdict against the pre-registered pass and kill thresholds (pass, kill, inconclusive, or failed), what it changes about the thesis claims, and two or three concrete follow-up experiments with expected information value. Also append the follow-ups to ${BACKLOG} under a heading 'Proposed by the loop' with ids L<iteration><letter>. Return the verdict and follow-ups.`, { label: `analyze:${plan.id}`, phase: 'Analyze', schema: SCHEMA_ANALYSIS, model: 'sonnet' })

  phase('Verify')
  const critic = await agent(`${RULES}\n\nYou are the adversarial critic for experiment ${plan.id}. Read the ledger entry in ${LEDGER} and the raw files under ${ROOT}/runs/${plan.id}/ and the script ${ROOT}/exp/${plan.id}.py if present. Check: every number in the entry matches the raw files; the verdict follows from the pre-registered thresholds (${plan.pass_threshold} / ${plan.kill_threshold}); no leakage (gold answer reaching the proxy or the allocator features); budget accounting equal across policies; sample size and seeds adequate for the claimed margin; no silent truncation or crash hidden in the logs. If anything is wrong, append a 'Critic note' paragraph under the entry in ${LEDGER} with the correction and the corrected verdict. Return ok=false if you corrected anything.`, { label: `critic:${plan.id}`, phase: 'Verify', schema: SCHEMA_CRITIC, model: 'sonnet' })
  done.push({ id: plan.id, machine: plan.machine, verdict: analysis ? analysis.verdict : 'failed', critic_ok: critic ? critic.ok : false, corrected: critic ? critic.corrected_verdict : '', one_line: analysis ? analysis.one_line : '' })
  log(`${plan.id}: ${analysis ? analysis.verdict : 'failed'}${critic && !critic.ok ? ' (critic corrected: ' + critic.corrected_verdict + ')' : ''}`)
}
return { iterations: done.length, results: done }
