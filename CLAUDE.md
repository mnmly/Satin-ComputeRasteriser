# Satin-ComputeRasteriser

A Metal compute-shader point-cloud rasterizer for Satin, with a streaming
out-of-core path for COPC datasets via SwiftPDAL.

## Build & run

Library target: `SatinComputeRasteriser`. Example app:
`Examples/SatinComputeRasteriserApp/SatinComputeRasteriserApp.xcodeproj`,
scheme `SatinComputeRasteriserApp` (Release for any perf measurement;
Debug builds are dominated by retain/release noise that doesn't exist in
Release and will mislead profiling).

Streaming depends on a patched `lazperf` shipped in
`copclib-1.2.0.xcframework` (the package pins
[`SwiftPDAL` 1.7.1+](https://github.com/mnmly/SwiftPDAL/releases/tag/1.7.1)).
No app-side action required; the optimization sits behind the existing
`CxxCOPC` C bridge.

## Documentation

`SatinComputeRasteriser` ships DocC-generated reference docs (see
`Sources/SatinComputeRasteriser/SatinComputeRasteriser.docc/` and
`Scripts/build_docs.sh`).
**`///` doc comments on `public` / `open` symbols are published** to
the static site (eventual target:
`https://mnmly.github.io/Satin-ComputeRasteriser/`) and, when
`EMIT_LLMS_TXT=1` is used, concatenated into `docs/llms.txt` for agent
consumption.

When you add or modify a `public` or `open` declaration:

- Write a `///` doc comment. One-sentence summary, then a paragraph if
  the *why* is non-obvious. Skip restating what the signature already
  says.
- Document each parameter with `- Parameter name:` (use the **internal**
  name when there's an external label — DocC warns otherwise).
- Cross-reference related symbols with double-backtick links, e.g.
  `` ``ComputeRasteriserPointCloud/addBatches(positionsXYZLow:positionsXYZMed:positionsXYZHigh:colors:levels:batches:commit:)`` ``.
  DocC link syntax is signature-sensitive: `foo(_:)` and `foo(_:_:)`
  are different symbols.
- When you add a new top-level symbol that belongs in the curated
  sidebar, add it under the appropriate `## Topics` group in
  `Sources/SatinComputeRasteriser/SatinComputeRasteriser.docc/SatinComputeRasteriser.md`.
  Topics are organized by *user task*, not alphabetic order.

Verify before declaring documentation work done:

```bash
Scripts/build_docs.sh
```

Expect exit 0 and no new "doesn't exist at" or "external name used to
document parameter" warnings attributable to your changes. The
`Scripts/build_docs.sh preview` form starts a local server with live
reload.

## Perf measurement

Streaming-pipeline CPU is the dominant cost on real workloads. Profile
with Time Profiler on a **Release** binary, attach to a running app
(`xcrun xctrace record --template "Time Profiler" --attach <pid>
--time-limit 20s --output <name>.trace`), and reproduce the streaming
scenario in steady state. The current top-of-trace (post-v0.2.0) is no
longer decode — it's the per-frame compute pipeline.

If you want a node-size histogram of what the COPC reader is being
asked for, set `SWIFTPDAL_NODE_HISTOGRAM=1` in the env before launching
— SwiftPDAL prints a log-bucketed histogram on quit. Off by default,
zero cost when unset.

## Conventions inherited from `~/.claude/CLAUDE.md`

Track tasks with the built-in task tools (`TaskCreate`/`TaskUpdate`), not
files. `tasks/todo.md` is a frozen archive of past plans and measurement
results — read it for history, don't extend it. Don't create
planning/decision files unless explicitly asked. Don't add comments that just restate the
code; reserve `///` for the documentation contract above and `//` for
genuinely non-obvious *why*.
