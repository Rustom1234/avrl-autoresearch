#!/usr/bin/env python3
"""Memory guard for the shared lab machines: keeps my total memory on one host under a limit.

    nohup /usr/bin/python3 curation/task/mem_guard.py > /dev/null 2>&1 &     one per host
    pkill -u $USER -f "mem_guar[d]"                                           stops it

Every second it adds up the resident memory of all my processes on this host. If the total passes --limit-gb, or the
machine has less than --min-free-gb available while a curation script of mine holds more than 2 GB, it kills the
model-written curation scripts that are running (python processes whose script name ends with --pattern, with their
children). It never kills anything else. curate_runner.py already caps each script's process at 16 GB; this is the
backstop for a script that starts several processes. Standard library only, so the system python runs it.
"""
import argparse, os, signal, socket, subprocess, time

GB = 1048576  # kB


def table():
    """(pid, ppid, rss kB, command line) of every process of mine."""
    out = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,rss=,args="], capture_output=True, text=True).stdout
    rows = [l.split(None, 3) for l in out.splitlines()]
    return [(int(p[0]), int(p[1]), int(p[2]), p[3]) for p in rows if len(p) == 4]


def trees(rows, pattern):
    """{root pid: (rss kB of the tree, pids, command line)} for every python process running a script named *pattern."""
    kids, rss = {}, {}
    for pid, ppid, r, _ in rows:
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = r
    found = {}
    for pid, _, _, args in rows:
        a = args.split()
        if len(a) > 1 and "python" in a[0] and a[1].endswith(pattern) and pid != os.getpid():
            pids, todo = [], [pid]
            while todo:
                q = todo.pop()
                pids.append(q)
                todo += kids.get(q, [])
            found[pid] = (sum(rss.get(q, 0) for q in pids), pids, args)
    return found


def available():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit-gb", type=float, default=40.0, help="my total resident memory on this host")
    ap.add_argument("--min-free-gb", type=float, default=6.0, help="memory the machine must keep available")
    ap.add_argument("--pattern", default="curate_runner.py")
    ap.add_argument("--log", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs",
                                                  "mem_guard_%s.log" % socket.gethostname().split(".")[0]))
    a = ap.parse_args(argv)

    def log(msg):
        with open(a.log, "a") as f:
            f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))

    log("started: limit %.1f GB, keep %.1f GB available, pattern %s, pid %d" % (a.limit_gb, a.min_free_gb, a.pattern, os.getpid()))
    peak, last_beat, last_warn = 0.0, time.time(), 0.0
    while True:
        rows = table()
        total, avail = sum(r[2] for r in rows) / GB, available() / GB
        peak = max(peak, total)
        over, low = total > a.limit_gb, avail < a.min_free_gb
        if over or low:
            killed = False
            for root, (kb, pids, args) in sorted(trees(rows, a.pattern).items(), key=lambda x: -x[1][0]):
                if over or kb > 2 * GB:
                    for q in pids:
                        try:
                            os.kill(q, signal.SIGKILL)
                        except OSError:
                            pass
                    killed = True
                    log("KILLED script tree %d (%.1f GB, %d processes): my total %.1f GB, available %.1f GB: %s"
                        % (root, kb / GB, len(pids), total, avail, args[-200:]))
            if over and not killed and time.time() - last_warn > 60:
                last_warn = time.time()
                log("WARNING my total %.1f GB is over the limit and no curation script is running; nothing killed" % total)
        if time.time() - last_beat > 600:
            log("ok: my total now %.1f GB, peak since last line %.1f GB, available %.1f GB" % (total, peak, avail))
            peak, last_beat = 0.0, time.time()
        time.sleep(1)


if __name__ == "__main__":
    main()
