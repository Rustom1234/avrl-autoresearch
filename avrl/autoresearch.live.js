export const meta = {
  name: 'avrl-autoresearch-live',
  description: 'AR5: run several copies of the autoresearch loop in parallel, one per run tree and GPU host, with per-role model assignment and an optional cascaded critic',
  phases: [
    { title: 'Gate', detail: 'clock and stop-file check (Haiku)' },
    { title: 'Plan', detail: 'choose the next experiment from the backlog and ledger' },
    { title: 'Run', detail: 'implement and run it on the run tree and its GPU machine' },
    { title: 'Analyze', detail: 'write the ledger entry with verdict and follow-ups' },
    { title: 'Verify', detail: 'critic or screener cascade checks numbers, verdict and confounds' },
  ],
}

const A = args || {}
if (!A.runs || !A.start_epoch) throw new Error('args.runs and args.start_epoch are required')
const STOP_AFTER = A.stop_after_s || 26100
const MAX_ITERS = A.max_iters || 40

function rulesFor(BASE) {
  return `Ground rules. Karpathy: think before coding, simplest code that works, surgical changes, concrete success criteria. Never use em dashes anywhere. Write only under ${BASE}/ (new scripts go in ${BASE}/avrl/exp/, results in ${BASE}/avrl/runs/<exp_id>/). Never git push. Never read or write anything outside ${BASE}/ except the read-only shared environments and models named below. Do not ask about a plan review, proceed directly. Compute rules: this run owns one machine with a single NVIDIA RTX A4000 16 GB GPU (vLLM at gpu_memory_utilization 0.85 needs about 13.9 GB, so one GPU job at a time); you never ssh anywhere yourself; every GPU job or long-running job is launched ONLY with ${BASE}/bin/gpu_run.sh <run_dir> '<command>' (it runs the command on the GPU machine with the right environment and a 45-minute timeout, and writes <run_dir>/run.log, <run_dir>/exit_code and <run_dir>/done.flag when the command finishes), checked with ${BASE}/bin/gpu_status.sh (it must show under 2 GB used before you launch; wait if it does not), and killed with ${BASE}/bin/gpu_kill.sh <run_dir>. Poll in-band only: 'until [ -f <run_dir>/done.flag ]; do ls <run_dir> > /dev/null; sleep 30; done' inside Bash calls of at most 540 seconds, repeating the call as needed; never start background pollers; never stop issuing tool calls while a job runs. Hard cap 45 minutes per experiment (enforced by the helper; if the cap hits, record the experiment as failed). CPU-only analysis runs directly in your shell with /homes/rdubash/active-audit/.venv/bin/python (sentence-transformers, sklearn, pandas, transformers); vLLM generation scripts use /homes/rdubash/active-audit/.venv-judge/bin/python and must go through gpu_run.sh. Both environments and /homes/rdubash/active-audit/models are read-only; never pip install. Existing assets: ${BASE}/avrl/p0_rollouts.py (vLLM rollouts, majority-vote proxy vs gold), ${BASE}/avrl/p0_analyze.py (structure metrics and allocation replay via ${BASE}/runs/sim_generic.py), models under ${BASE}/models/, data under ${BASE}/avrl/data/, results under ${BASE}/avrl/runs/. Copy and modify scripts rather than editing p0 scripts in place. Report failures honestly with the exact error text.`
}

const SCHEMA_GATE = { type: 'object', properties: { epoch: { type: 'number' }, stop: { type: 'boolean' } }, required: ['epoch', 'stop'] }
const SCHEMA_PLAN = { type: 'object', properties: { id: { type: 'string' }, hypothesis: { type: 'string' }, plan: { type: 'string' }, command_shape: { type: 'string' }, metric: { type: 'string' }, pass_threshold: { type: 'string' }, kill_threshold: { type: 'string' }, expected_minutes: { type: 'number' }, machine: { type: 'string', enum: ['local', 'none'] }, why_now: { type: 'string' } }, required: ['id', 'hypothesis', 'plan', 'command_shape', 'metric', 'pass_threshold', 'kill_threshold', 'expected_minutes', 'machine', 'why_now'] }
const SCHEMA_RUN = { type: 'object', properties: { ok: { type: 'boolean' }, result_files: { type: 'array', items: { type: 'string' } }, scripts_written: { type: 'array', items: { type: 'string' } }, key_numbers: { type: 'string' }, wall_minutes: { type: 'number' }, error: { type: 'string' } }, required: ['ok', 'result_files', 'scripts_written', 'key_numbers', 'wall_minutes', 'error'] }
const SCHEMA_ANALYSIS = { type: 'object', properties: { verdict: { type: 'string', enum: ['pass', 'kill', 'inconclusive', 'failed'] }, one_line: { type: 'string' }, follow_ups: { type: 'array', items: { type: 'string' } }, thesis_impact: { type: 'string' }, claim2: { type: 'string', enum: ['supported', 'not_supported', 'untested'] }, claim3: { type: 'string', enum: ['supported', 'not_supported', 'untested'] } }, required: ['verdict', 'one_line', 'follow_ups', 'thesis_impact', 'claim2', 'claim3'] }
const SCHEMA_CRITIC = { type: 'object', properties: { ok: { type: 'boolean' }, issues: { type: 'array', items: { type: 'string' } }, corrected_verdict: { type: 'string' } }, required: ['ok', 'issues', 'corrected_verdict'] }
const SCHEMA_SCREEN = { type: 'object', properties: { ok: { type: 'boolean' }, issues: { type: 'array', items: { type: 'string' } }, verdict: { type: 'string', enum: ['pass', 'kill', 'inconclusive', 'failed'] } }, required: ['ok', 'issues', 'verdict'] }

async function runLoop(R) {
  const ID = R.run_id
  const BASE = R.base
  const M = R.models
  const ROOT = `${BASE}/avrl`
  const LEDGER = `${ROOT}/RESEARCH_LOG.md`
  const BACKLOG = `${BASE}/lit/avrl/EXPERIMENT_BACKLOG.md`
  const SYNTH = `${BASE}/lit/avrl/SYNTHESIS.md`
  const RULES = rulesFor(BASE)
  const done = []
  const trace = []
  let dry = 0
  let endReason = 'MAX_ITERS'
  for (let i = 0; i < MAX_ITERS && dry < 2; i++) {
    const n = i + 1
    const gate = await agent(`Run exactly one bash command: echo "$(date +%s) $(test -f ${BASE}/STOP && echo 1 || echo 0)" and return the first field as epoch (a number) and stop=true if the second field is 1, else false. Do nothing else.`, { label: `${ID} gate:${n}`, phase: 'Gate', schema: SCHEMA_GATE, model: 'haiku' })
    if (gate && (gate.stop || gate.epoch >= A.start_epoch + STOP_AFTER)) { endReason = gate.stop ? 'STOP_FILE' : 'DEADLINE'; log(`${ID}: stopping before iteration ${n} (${endReason})`); break }

    const plan = await agent(`${RULES}\n\nIteration ${n} of at most ${MAX_ITERS}. Read ${BACKLOG}, ${SYNTH} (skim), ${ROOT}/PROJECT.md, ${LEDGER} if it exists (it lists what has been run and the follow-ups proposed), and the newest result json files under ${ROOT}/runs/ (ls -t). Pick the single next experiment with the highest information value per GPU hour for the thesis claims that is runnable NOW on this machine in under 45 minutes. Do not repeat an experiment already in the ledger unless the ledger asks for a replication with more seeds. Prefer experiments that can kill a claim over ones that can only confirm it. If nothing runnable remains, return id NONE and say why. Already run this loop: ${JSON.stringify(done)}.`, { label: `${ID} plan:${n}`, phase: 'Plan', schema: SCHEMA_PLAN, model: M.plan })
    if (!plan || plan.id === 'NONE' || plan.machine === 'none') { dry++; log(`${ID} iteration ${n}: nothing runnable (${plan ? plan.why_now : 'planner died'})`); trace.push({ iter: n, gate, plan, skipped: true }); if (dry >= 2) endReason = 'DRY'; continue }
    dry = 0
    log(`${ID} iteration ${n}: running ${plan.id}: ${plan.hypothesis}`)

    const run = await agent(`${RULES}\n\nImplement and run experiment ${plan.id}. Spec: ${JSON.stringify(plan)}. Write any new script to ${ROOT}/exp/${plan.id}.py (or .sh), results to ${ROOT}/runs/${plan.id}/ including a result.json with the metric values and a done.flag when finished. Reuse the p0 scripts by copying; keep changes minimal. Before launching a GPU job, check the GPU is free with gpu_status.sh; launch it only through gpu_run.sh, poll until done.flag appears, cap at 45 minutes. If it crashes, read the traceback, fix once, rerun once; if it fails again return ok=false with the exact error. When done, return the result file paths and a compact excerpt of the key numbers (the metric, the baselines, sample sizes).`, { label: `${ID} run:${n}`, phase: 'Run', schema: SCHEMA_RUN, model: M.run })
    if (!run || !run.ok) { log(`${ID} ${plan.id} failed: ${run ? run.error : 'runner died'}`) }

    const analysis = await agent(`${RULES}\n\nExperiment ${plan.id} finished. Spec: ${JSON.stringify(plan)}. Run report: ${JSON.stringify(run)}. Read the result files and logs under ${ROOT}/runs/${plan.id}/ yourself, do not trust the run report's numbers without checking. Append an entry to ${LEDGER} (create the file with a short header if it does not exist) in this exact shape: '## <id> <date from the bash date command> <verdict>' then hypothesis, what was run (model, task, n, seeds, budget), a markdown table of the numbers with baselines, verdict against the pre-registered pass and kill thresholds (pass, kill, inconclusive, or failed), what it changes about the thesis claims, and two or three concrete follow-up experiments with expected information value. Also append the follow-ups to ${BACKLOG} under a heading 'Proposed by the loop' with ids L<iteration><letter>. Return the verdict and follow-ups, and your current reading of claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget) and claim 3 (the edge comes from structure in prompt space) as supported, not_supported or untested.`, { label: `${ID} analyze:${n}`, phase: 'Analyze', schema: SCHEMA_ANALYSIS, model: M.analyze })

    const criticPrompt = `${RULES}\n\nYou are the adversarial critic for experiment ${plan.id}. Read the ledger entry in ${LEDGER} and the raw files under ${ROOT}/runs/${plan.id}/ and the script ${ROOT}/exp/${plan.id}.py if present. Check: every number in the entry matches the raw files; the verdict follows from the pre-registered thresholds (${plan.pass_threshold} / ${plan.kill_threshold}); no leakage (gold answer reaching the proxy or the allocator features); budget accounting equal across policies; sample size and seeds adequate for the claimed margin; no silent truncation or crash hidden in the logs. If anything is wrong, append a 'Critic note' paragraph under the entry in ${LEDGER} with the correction and the corrected verdict. Return ok=false if you corrected anything.`
    let critic = null
    let cascade = null
    if (R.critic_mode === 'cascade') {
      const screenPrompt = `${RULES}\n\nYou are a read-only screener for experiment ${plan.id}. Read the ledger entry in ${LEDGER} and the raw files under ${ROOT}/runs/${plan.id}/ and the script ${ROOT}/exp/${plan.id}.py if present. Check: every number in the entry matches the raw files; the verdict follows from the pre-registered thresholds (${plan.pass_threshold} / ${plan.kill_threshold}); no leakage (gold answer reaching the proxy or the allocator features); budget accounting equal across policies; sample size and seeds adequate for the claimed margin; no silent truncation or crash hidden in the logs. Do NOT write to any file. Return ok=true if the entry is correct as written, otherwise ok=false with the problems in issues, and in verdict the verdict the numbers actually support.`
      const screens = await parallel([0, 1].map(j => () => agent(screenPrompt, { label: `${ID} screen${j}:${n}`, phase: 'Verify', schema: SCHEMA_SCREEN, model: M.screen })))
      const [s0, s1] = screens
      const why = (!s0 || !s1) ? 'null' : (!s0.ok || !s1.ok) ? 'flag' : (s0.verdict !== s1.verdict) ? 'disagree' : null
      cascade = { s0, s1, escalated: !!why, why }
      if (why) {
        critic = await agent(criticPrompt, { label: `${ID} critic:${n}`, phase: 'Verify', schema: SCHEMA_CRITIC, model: M.critic })
      } else {
        critic = { ok: true, issues: [], corrected_verdict: '' }
      }
    } else {
      critic = await agent(criticPrompt, { label: `${ID} critic:${n}`, phase: 'Verify', schema: SCHEMA_CRITIC, model: M.critic })
    }

    done.push({ id: plan.id, machine: plan.machine, verdict: analysis ? analysis.verdict : 'failed', critic_ok: critic ? critic.ok : false, corrected: critic ? critic.corrected_verdict : '', one_line: analysis ? analysis.one_line : '' })
    trace.push({ iter: n, gate, plan, run, analysis, critic, cascade })
    log(`${ID} ${plan.id}: ${analysis ? analysis.verdict : 'failed'}${critic && !critic.ok ? ' (critic corrected: ' + critic.corrected_verdict + ')' : ''}${cascade ? ' [cascade ' + (cascade.escalated ? 'escalated: ' + cascade.why : 'not escalated') + ']' : ''}`)
  }
  return { run_id: ID, iterations: done.length, end_reason: endReason, done, trace }
}

const results = await parallel(A.runs.map(R => () => runLoop(R)))
return { runs: results.filter(Boolean) }
