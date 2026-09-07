#!/usr/bin/env python3
"""Aggregate a DS4_METAL_ENCODER_TIMELINE log per kernel.

Each timestamped encoder holds one dispatch when the timeline is active, so
the log attributes GPU time to individual kernels.  Batches (command buffers)
whose total duration exceeds --max-batch-ms are treated as prefill and
skipped, leaving the decode tokens; per-kernel totals are then divided by the
number of decode batches so the table reads as microseconds per token.
"""
import argparse
import collections
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--max-batch-ms", type=float, default=200.0)
    ap.add_argument("--skip-batches", type=int, default=2, help="warm-up batches to ignore")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--batches-per-token", type=int, default=2,
                    help="command buffers per decode token (flush at layer 2 gives two)")
    a = ap.parse_args()
    batches = collections.OrderedDict()
    for line in open(a.log):
        if not line.startswith("E "):
            continue
        f = line.split()
        seq, idx = int(f[1]), int(f[2])
        dur, gap = float(f[5]), float(f[6])
        kernel = f[11] if len(f) > 11 else "?"
        tg, tpt = f[9], f[10]
        batches.setdefault(seq, []).append((idx, dur, gap, kernel, tg, tpt))
    decode = [(seq, recs) for seq, recs in batches.items()
              if sum(r[1] for r in recs) < a.max_batch_ms * 1000.0]
    decode = decode[a.skip_batches:]
    if not decode:
        sys.exit("no decode batches found")
    n_tokens = max(1, len(decode) // a.batches_per_token)
    per = collections.defaultdict(lambda: [0, 0.0, 0.0, set()])
    total_dur = total_gap = 0.0
    n_dispatch = 0
    for _, recs in decode:
        for _, dur, gap, kernel, tg, tpt in recs:
            p = per[kernel]
            p[0] += 1
            p[1] += dur
            p[2] += gap
            p[3].add(f"{tg}/{tpt}")
            total_dur += dur
            total_gap += gap
            n_dispatch += 1
    print(f"decode batches: {len(decode)} (~{n_tokens} tokens), dispatches/token: {n_dispatch / n_tokens:.0f}, "
          f"gpu busy/token: {total_dur / n_tokens / 1000:.3f} ms, gaps/token: {total_gap / n_tokens / 1000:.3f} ms")
    print(f"{'kernel':<56} {'n/tok':>6} {'us/tok':>9} {'%':>6} {'us/disp':>8} {'gap/tok':>8}  grids")
    rows = sorted(per.items(), key=lambda kv: -kv[1][1])
    for kernel, (n, dur, gap, grids) in rows[: a.top]:
        print(f"{kernel:<56} {n / n_tokens:>6.1f} {dur / n_tokens:>9.1f} {100 * dur / total_dur:>6.2f} "
              f"{dur / n:>8.2f} {gap / n_tokens:>8.1f}  {' '.join(sorted(grids))[:60]}")


if __name__ == "__main__":
    main()
