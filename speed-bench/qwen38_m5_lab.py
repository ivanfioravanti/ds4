#!/usr/bin/env python3
"""Qwen3.8 Metal optimization lab: exactness gate plus prefill/decode measurement.

Compares a frozen baseline build (a separate worktree with its own binaries and
metal/ sources) against the working tree.  Every measurement is interleaved
and every accepted change must keep full-vocabulary FP32 logits byte-identical.

Subcommands:
  parity     teacher-forced full-logit parity (1..128 prompt tokens, 32 rows each)
  sweep      per-frontier ds4-bench prefill+greedy-decode runs, four fresh
             processes per frontier in ABBA/BAAB order, frontier logits
             compared byte for byte across all four
  decode     short-prompt plain decode and MTP decode (qwen38_mtp_compare.py)
  mtp-long   MTP decode on long raw prompts (32K/128K), both builds interleaved
  prefill-ab single-engine env A/B (metal_prefill_variant_bench)
  decode-ab  single-engine env A/B (qwen38_decode_variant_bench)
  gate       parity + decode + sweep, then a summary JSON and Markdown

Raw artifacts go under --out (default: OUT/qwen38-m5-lab/<name>, ignored by
git); commit only the summary files.
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT.parent / "ds4-baseline"
DEFAULT_MODEL = ROOT / "gguf" / ("Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-"
                                 "Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf")
DEFAULT_PLE = ROOT / "gguf" / "Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
DEFAULT_PROMPT = ROOT / "speed-bench" / "promessi_sposi.txt"
RATE = re.compile(r"ds4: (?:Qwen3\.8 )?prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s")
ACCEPT = re.compile(r"ds4: Qwen3\.8 mtp: (\d+) verify cycles, (\d+) drafts accepted")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
    if extra:
        env.update(extra)
    return env


def git_head(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "?"


class Lab:
    def __init__(self, args):
        self.args = args
        self.base = Path(args.base).resolve()
        self.cand = ROOT
        self.model = Path(args.model).resolve()
        self.ple = Path(args.ple).resolve()
        self.prompt = Path(args.prompt).resolve()
        self.out = Path(args.out).resolve() if args.out else ROOT / "OUT" / "qwen38-m5-lab" / args.name
        self.out.mkdir(parents=True, exist_ok=True)
        self.builds = {"baseline": self.base, "candidate": self.cand}
        self.log = (self.out / "lab.log").open("a")
        self.report = {
            "name": args.name, "started_utc": datetime.now(timezone.utc).isoformat(),
            "model": str(self.model), "ple": str(self.ple), "prompt": str(self.prompt),
            "builds": {name: {"dir": str(d), "head": git_head(d),
                              "ds4_sha256": sha256(d / "ds4") if (d / "ds4").exists() else None,
                              "ds4_bench_sha256": sha256(d / "ds4-bench") if (d / "ds4-bench").exists() else None,
                              "metal_sha256": {p.name: sha256(p) for p in sorted((d / "metal").glob("*.metal"))}}
                       for name, d in self.builds.items()},
            "candidate_env": dict(kv.split("=", 1) for kv in args.candidate_env),
            "steps": {},
        }
        differing = [n for n in self.report["builds"]["baseline"]["metal_sha256"]
                     if self.report["builds"]["baseline"]["metal_sha256"][n] !=
                     self.report["builds"]["candidate"]["metal_sha256"].get(n)]
        self.report["metal_sources_differing"] = differing
        self.say(f"lab {args.name}: baseline {self.report['builds']['baseline']['head']} vs candidate "
                 f"{self.report['builds']['candidate']['head']}; differing metal sources: {differing or 'none'}")

    def say(self, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def save(self):
        (self.out / "results.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def env_for(self, build):
        return clean_env(self.report["candidate_env"] if build == "candidate" else None)

    def run(self, build, cmd, stem, cwd=None, env=None):
        """Run cmd from the build directory so that build's metal/ sources load."""
        cwd = cwd or self.builds[build]
        stem = Path(stem)
        t0 = time.monotonic()
        with stem.with_suffix(".stdout").open("wb") as out, stem.with_suffix(".stderr").open("wb") as err:
            rc = subprocess.run(cmd, cwd=cwd, env=env or self.env_for(build), stdout=out, stderr=err).returncode
        wall = time.monotonic() - t0
        if rc != 0:
            tail = stem.with_suffix(".stderr").read_text()[-2000:]
            raise RuntimeError(f"{' '.join(map(str, cmd))} failed rc={rc} in {wall:.1f}s:\n{tail}")
        return wall

    # ---------------------------------------------------------------- parity
    def parity(self):
        """Teacher-forced full-logit dumps through the production Metal graph."""
        d = self.out / "parity"
        d.mkdir(exist_ok=True)
        text = [814, 20139, 1204, 264, 6165, 20653, 264, 3370, 1622, 888,
                25804, 321, 20539, 279, 1965, 13]
        sizes = [1, 2, 8, 9, 39, 128]
        rec = {"sizes": sizes, "checks": []}
        for build, bdir in self.builds.items():
            listing = d / f"{build}.tsv"
            listing.write_text("".join(
                ",".join(map(str, (text * 8)[:n])) + "|" + ",".join(map(str, text * 2)) +
                "\t" + str((d / f"{build}-{n}.f32").resolve()) + "\n" for n in sizes))
            env = self.env_for(build)
            env.update({"DS4_QWEN4_FT_LIST": str(listing), "DS4_QWEN4_GPU": "1"})
            cmd = [str(bdir / "ds4"), "-m", str(self.model), "--ple", str(self.ple), "--metal",
                   "-c", "32768", "--nothink", "--temp", "0", "--first-token-test", "--raw", "-p", "x"]
            wall = self.run(build, cmd, d / f"{build}-ft", env=env)
            self.say(f"parity {build}: {wall:.1f}s")
        ok = True
        for n in sizes:
            paths = [d / f"{b}-{n}.f32" for b in self.builds]
            hashes = [sha256(p) for p in paths]
            same = hashes[0] == hashes[1] and paths[0].stat().st_size == paths[1].stat().st_size > 0
            ok &= same
            rec["checks"].append({"prompt_tokens": n, "rows": 32, "bytes": paths[0].stat().st_size,
                                  "exact": same, "sha256": hashes})
            self.say(f"parity n={n}: {'EXACT' if same else 'MISMATCH'}")
        rec["exact"] = ok
        self.report["steps"]["parity"] = rec
        self.save()
        return ok

    # ----------------------------------------------------------------- sweep
    def sweep_one(self, build, tag, ctx):
        """One frontier in one process: prefill ctx tokens, then greedy decode."""
        d = self.out / "sweep" / f"{tag}-{build}"
        (d / "logits").mkdir(parents=True, exist_ok=True)
        gen = self.args.gen_tokens
        cmd = [str(self.builds[build] / "ds4-bench"), "-m", str(self.model), "--ple", str(self.ple), "--metal",
               "--prompt-file", str(self.prompt), "--ctx-start", str(ctx), "--ctx-max", str(ctx),
               "--gen-tokens", str(gen), "--csv", str(d / "speed.csv"),
               "--dump-frontier-logits-dir", str(d / "logits")]
        if self.args.prefill_chunk:
            cmd += ["--prefill-chunk", str(self.args.prefill_chunk)]
        env = self.env_for(build)
        env["DS4_BENCH_FORCE_SNAPSHOT"] = "1"
        wall = self.run(build, cmd, d / "bench", env=env)
        f = (d / "speed.csv").read_text().splitlines()[1].split(",")
        row = {"ctx": int(f[0]), "prefill_tokens": int(f[1]), "prefill_tps": float(f[2]),
               "gen_tps": float(f[4]), "gen_first_ms": float(f[5]), "gen_steady_tps": float(f[7])}
        logits = {p.name: sha256(p) for p in sorted((d / "logits").glob("frontier_*.logits.json"))}
        self.say(f"sweep {tag} {build}: {wall:.0f}s prefill {row['prefill_tps']:.1f} decode {row['gen_tps']:.2f}")
        return {"row": row, "logits_sha256": logits, "wall_s": wall}

    def sweep_ctxs(self):
        if self.args.sweep_ctx:
            return [int(c) for c in self.args.sweep_ctx.split(",")]
        ctxs, c = [], self.args.ctx_start
        while c < self.args.ctx_max:
            ctxs.append(c)
            c *= 2
        return ctxs + [self.args.ctx_max]

    def sweep(self):
        """Per frontier, four fresh processes in ABBA (or BAAB) order.

        Sustained GPU load drifts the machine by 10-20% over a few minutes,
        so builds are interleaved within each frontier rather than run as two
        long sweeps; every frontier's logits must match across all four runs.
        """
        runs, table, exact = {}, [], True
        for i, ctx in enumerate(self.sweep_ctxs()):
            pattern = ["baseline", "candidate", "candidate", "baseline"]
            if i % 2:
                pattern.reverse()
            per = {b: [] for b in self.builds}
            hashes = set()
            for slot, build in enumerate(pattern):
                r = self.sweep_one(build, f"c{ctx}-s{slot}", ctx)
                runs[f"c{ctx}-s{slot}-{build}"] = r
                per[build].append(r["row"])
                hashes.update(r["logits_sha256"].values())
            entry = {"ctx": ctx, "exact": len(hashes) == 1}
            exact &= entry["exact"]
            for b, rows in per.items():
                entry[b] = {"prefill_tps": statistics.mean(v["prefill_tps"] for v in rows),
                            "gen_tps": statistics.mean(v["gen_tps"] for v in rows),
                            "gen_steady_tps": statistics.mean(v["gen_steady_tps"] for v in rows)}
            entry["prefill_pct"] = (entry["candidate"]["prefill_tps"] / entry["baseline"]["prefill_tps"] - 1) * 100
            entry["decode_pct"] = (entry["candidate"]["gen_tps"] / entry["baseline"]["gen_tps"] - 1) * 100
            table.append(entry)
            self.say(f"sweep ctx={ctx}: prefill {entry['baseline']['prefill_tps']:.1f} -> "
                     f"{entry['candidate']['prefill_tps']:.1f} ({entry['prefill_pct']:+.2f}%), decode "
                     f"{entry['baseline']['gen_tps']:.2f} -> {entry['candidate']['gen_tps']:.2f} "
                     f"({entry['decode_pct']:+.2f}%), logits {'EXACT' if entry['exact'] else 'MISMATCH'}")
            rec = {"gen_tokens": self.args.gen_tokens, "runs": runs, "table": table,
                   "frontier_logits_exact": exact, "frontiers_compared": len(table)}
            self.report["steps"]["sweep"] = rec
            self.save()
        self.say(f"sweep frontier logits: {'EXACT' if exact else 'MISMATCH'} over {len(table)} frontiers")
        return exact

    # ---------------------------------------------------------------- decode
    def decode(self):
        """Short-prompt plain and MTP decode through the CLI compare harness."""
        rec = {}
        for mode in ("plain", "mtp"):
            d = self.out / "decode" / mode
            cmd = [sys.executable, str(ROOT / "speed-bench" / "qwen38_mtp_compare.py"),
                   "--model", str(self.model), "--ple", str(self.ple),
                   "--baseline", str(self.base / "ds4"),
                   "--baseline-source", str(self.base / "metal" / "qwen4.metal"),
                   "--baseline-moe-source", str(self.base / "metal" / "moe.metal"),
                   "--candidate", str(self.cand / "ds4"),
                   "--candidate-source", str(self.cand / "metal" / "qwen4.metal"),
                   "--candidate-moe-source", str(self.cand / "metal" / "moe.metal"),
                   "--ctx", "8192", "--repeats", str(self.args.decode_repeats), "--out", str(d)]
            for kv in self.args.candidate_env:
                cmd += ["--candidate-env", kv]
            if mode == "plain":
                cmd.append("--no-mtp")
            d.mkdir(parents=True, exist_ok=True)
            wall = self.run("candidate", cmd, d / "compare", cwd=ROOT, env=clean_env())
            res = json.loads((d / "results.json").read_text())
            rec[mode] = {"summary": res["summary"], "wall_s": wall}
            for case, s in res["summary"].items():
                self.say(f"decode {mode} {case}: {s['median_tps']['baseline']:.2f} -> "
                         f"{s['median_tps']['candidate']:.2f} ({(s['speedup'] - 1) * 100:+.2f}%) "
                         f"outputs {'identical' if s['all_outputs_identical'] else 'DIFFER'}")
        self.report["steps"]["decode"] = rec
        self.save()
        return all(s["all_outputs_identical"] and s["acceptance_identical"]
                   for m in rec.values() for s in m["summary"].values())

    # -------------------------------------------------------------- mtp-long
    def mtp_long(self):
        """MTP decode after long raw prompts, both builds interleaved."""
        d = self.out / "mtp-long"
        d.mkdir(exist_ok=True)
        text = self.prompt.read_text(errors="ignore")
        rec = {}
        for chars in self.args.long_chars:
            slice_path = d / f"prompt-{chars}.txt"
            slice_path.write_text(text[:chars])
            runs = []
            for rep in range(self.args.long_repeats):
                for build in (["baseline", "candidate"] if rep % 2 == 0 else ["candidate", "baseline"]):
                    stem = d / f"{chars}-{build}-r{rep}"
                    cmd = [str(self.builds[build] / "ds4"), "-m", str(self.model), "--ple", str(self.ple),
                           "--metal", "--ctx", str(self.args.long_ctx), "-n", str(self.args.long_tokens),
                           "--temp", "0", "--nothink", "--mtp", "--mtp-timing",
                           "--prompt-file", str(slice_path), "--raw-prompt"]
                    wall = self.run(build, cmd, stem)
                    log = stem.with_suffix(".stderr").read_text()
                    rates, accepts = RATE.findall(log), ACCEPT.findall(log)
                    r = {"build": build, "rep": rep, "prefill_tps": float(rates[-1][0]),
                         "decode_tps": float(rates[-1][1]),
                         "cycles": int(accepts[-1][0]) if accepts else 0,
                         "accepted": int(accepts[-1][1]) if accepts else 0,
                         "output_sha256": sha256(stem.with_suffix(".stdout")), "wall_s": wall}
                    runs.append(r)
                    self.say(f"mtp-long {chars} {build} r{rep}: prefill {r['prefill_tps']:.1f} decode "
                             f"{r['decode_tps']:.2f} accepted {r['accepted']}/{r['cycles']}")
            med = {b: statistics.median(r["decode_tps"] for r in runs if r["build"] == b) for b in self.builds}
            rec[str(chars)] = {"runs": runs, "median_decode_tps": med,
                               "decode_pct": (med["candidate"] / med["baseline"] - 1) * 100,
                               "outputs_identical": len({r["output_sha256"] for r in runs}) == 1,
                               "acceptance_identical": len({(r["accepted"], r["cycles"]) for r in runs}) == 1}
            self.say(f"mtp-long {chars}: {med['baseline']:.2f} -> {med['candidate']:.2f} "
                     f"({rec[str(chars)]['decode_pct']:+.2f}%) outputs "
                     f"{'identical' if rec[str(chars)]['outputs_identical'] else 'DIFFER'}")
        self.report["steps"]["mtp_long"] = rec
        self.save()
        return all(v["outputs_identical"] and v["acceptance_identical"] for v in rec.values())

    # ---------------------------------------------------------------- env A/B
    def prefill_ab(self):
        d = self.out / "prefill-ab"
        d.mkdir(exist_ok=True)
        name, _, value = self.args.ab_env.partition("=")
        cmd = [str(self.cand / "speed-bench" / "metal_prefill_variant_bench"), "-m", str(self.model),
               "--ple", str(self.ple), "--prompt-file", str(self.prompt),
               "--prefill-chunk", str(self.args.ab_chunk), "--prefix-tokens", str(self.args.ab_prefix),
               "--warmup-tokens", str(self.args.ab_warmup), "--candidate-env", name,
               "--candidate-value", value or "1", "--repeats", str(self.args.ab_repeats)]
        if self.args.ab_initial:
            cmd += ["--initial-tokens", str(self.args.ab_initial)]
        wall = self.run("candidate", cmd, d / "run", cwd=ROOT)
        out = (d / "run.stdout").read_text()
        self.say(out.strip().splitlines()[-1] if out.strip() else "no output")
        self.report["steps"]["prefill_ab"] = {"env": self.args.ab_env, "stdout": out, "wall_s": wall}
        self.save()

    def decode_ab(self):
        d = self.out / "decode-ab"
        d.mkdir(exist_ok=True)
        cmd = [str(self.cand / "speed-bench" / "qwen38_decode_variant_bench"), "-m", str(self.model),
               "--ple", str(self.ple), "--prompt-file", str(self.prompt),
               "--prefix-tokens", str(self.args.ab_prefix), "--warmup", str(self.args.ab_warmup),
               "--tokens", str(self.args.ab_tokens)]
        for kv in self.args.ab_env.split(","):
            cmd += ["--candidate-env", kv]
        for kv in self.args.ab_control_env:
            cmd += ["--control-env", kv]
        wall = self.run("candidate", cmd, d / "run", cwd=ROOT)
        out = (d / "run.stdout").read_text()
        for line in out.strip().splitlines():
            self.say(line)
        self.report["steps"]["decode_ab"] = {"env": self.args.ab_env, "stdout": out, "wall_s": wall}
        self.save()

    # ------------------------------------------------------------------ gate
    def gate(self):
        ok = self.parity()
        if not ok and not self.args.keep_going:
            self.say("parity FAILED; stopping the gate")
            self.summary(False)
            return False
        ok &= self.decode()
        ok &= self.sweep()
        if self.args.long_chars:
            ok &= self.mtp_long()
        self.summary(ok)
        return ok

    def summary(self, ok):
        s = self.report["steps"]
        lines = [f"# Lab {self.args.name}", "",
                 f"baseline `{self.report['builds']['baseline']['head']}` vs candidate "
                 f"`{self.report['builds']['candidate']['head']}`; candidate env "
                 f"`{self.report['candidate_env'] or 'none'}`; exact: **{'yes' if ok else 'NO'}**", ""]
        if "parity" in s:
            lines += [f"- parity: {'EXACT' if s['parity']['exact'] else 'MISMATCH'} on "
                      f"{len(s['parity']['checks'])} teacher-forced histories x 32 rows"]
        if "decode" in s:
            for mode, m in s["decode"].items():
                for case, v in m["summary"].items():
                    lines.append(f"- decode {mode} {case}: {v['median_tps']['baseline']:.2f} -> "
                                 f"{v['median_tps']['candidate']:.2f} t/s ({(v['speedup'] - 1) * 100:+.2f}%)")
        if "sweep" in s:
            lines += ["", "| ctx | prefill base | prefill cand | prefill % | decode base | decode cand | decode % |",
                      "|---:|---:|---:|---:|---:|---:|---:|"]
            for e in s["sweep"]["table"]:
                lines.append(f"| {e['ctx']} | {e['baseline']['prefill_tps']:.1f} | {e['candidate']['prefill_tps']:.1f} "
                             f"| {e['prefill_pct']:+.2f} | {e['baseline']['gen_tps']:.2f} | "
                             f"{e['candidate']['gen_tps']:.2f} | {e['decode_pct']:+.2f} |")
            lines.append(f"\nfrontier logits: {'EXACT' if s['sweep']['frontier_logits_exact'] else 'MISMATCH'} "
                         f"over {s['sweep']['frontiers_compared']} frontiers")
        if "mtp_long" in s:
            for chars, v in s["mtp_long"].items():
                lines.append(f"- mtp-long {chars} chars: {v['median_decode_tps']['baseline']:.2f} -> "
                             f"{v['median_decode_tps']['candidate']:.2f} t/s ({v['decode_pct']:+.2f}%)")
        (self.out / "summary.md").write_text("\n".join(lines) + "\n")
        self.report["exact"] = ok
        self.save()
        self.say("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["parity", "sweep", "decode", "mtp-long", "prefill-ab", "decode-ab", "gate"])
    ap.add_argument("--name", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--out")
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--ple", default=str(DEFAULT_PLE))
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    ap.add_argument("--candidate-env", action="append", default=[], metavar="NAME=VALUE",
                    help="environment applied to the candidate build in parity/sweep/decode/mtp-long")
    ap.add_argument("--ctx-start", type=int, default=4096)
    ap.add_argument("--ctx-max", type=int, default=131072)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--sweep-ctx", default="", help="comma-separated frontiers (default: doubling ctx-start..ctx-max)")
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--decode-repeats", type=int, default=3)
    ap.add_argument("--long-chars", type=int, action="append", default=[], help="raw prompt slice sizes for mtp-long")
    ap.add_argument("--long-ctx", type=int, default=140000)
    ap.add_argument("--long-tokens", type=int, default=256)
    ap.add_argument("--long-repeats", type=int, default=2)
    ap.add_argument("--ab-env", default="", help="NAME[=VALUE][,NAME=VALUE...] for prefill-ab/decode-ab")
    ap.add_argument("--ab-control-env", action="append", default=[])
    ap.add_argument("--ab-prefix", type=int, default=8192)
    ap.add_argument("--ab-initial", type=int, default=0)
    ap.add_argument("--ab-chunk", type=int, default=8192)
    ap.add_argument("--ab-warmup", type=int, default=32)
    ap.add_argument("--ab-repeats", type=int, default=2)
    ap.add_argument("--ab-tokens", type=int, default=512)
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()
    lab = Lab(args)
    ok = {"parity": lab.parity, "sweep": lab.sweep, "decode": lab.decode, "mtp-long": lab.mtp_long,
          "prefill-ab": lab.prefill_ab, "decode-ab": lab.decode_ab, "gate": lab.gate}[args.command]()
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
