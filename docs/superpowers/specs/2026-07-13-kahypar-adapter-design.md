# KaHyPar 1.3.7 clean-mainline adapter design

## Scope

Configure the existing clean-mainline KaHyPar partition-hyperedge path for the
server's installed `kahypar==1.3.7` Python package. This change must not alter
DAG, k-hop, or attribute hyperedge construction, PPO, rewards, or environment
dynamics.

## Dependency and configuration asset

The training server uses CPython 3.13 on x86-64 Linux and has the official
`kahypar==1.3.7` manylinux wheel installed in the `uav322` environment. The
wheel contains the Python extension but does not bundle an algorithm INI file.

Vendor the upstream KaHyPar `cut_rKaHyPar_sea20.ini` configuration under:

`third_party/kahypar/cut_rKaHyPar_sea20.ini`

Keep its upstream license/attribution header and record the source URL in the
project documentation. The repository copy makes local, GitHub, and server
behavior reproducible and avoids a machine-specific absolute path.

## Resource resolution

Add clean-mainline configuration values for:

- the repository-relative KaHyPar INI path;
- a deterministic KaHyPar seed of `0`;
- a worker-response timeout of `10` seconds.

Resolve the INI path with a small, testable repository-root locator that walks
upward from the module location until one directory contains both `config.py`
and `environment/graph_builder.py`. This marker pair also works in a deployed
source tree without `.git`. Do not depend on the current working directory, a
machine-specific absolute path, or a fixed number of `parents[...]` hops. A
missing or unreadable INI is a configuration failure and must produce the
existing degraded status instead of reporting success.

## Process architecture

KaHyPar must not run in the training process. Each `CleanGraphBuilder` lazily
owns one persistent worker process, started with Python's `spawn`
multiprocessing context. The parent and worker exchange one request and response
at a time over a pipe. A request contains only the node count, cleaned base
hyperedges, partition count, epsilon, seed, and resolved INI path. The worker is
a minimal module that does not import Torch or the training stack.

The parent serializes calls with a lock. Current clean training and evaluation
are single-environment synchronous loops, but the lock prevents future
same-process threads from concurrently driving the native worker protocol.

For every request, the worker must:

1. construct the `kahypar.Context`;
2. load the vendored INI with `loadINIconfiguration`;
3. set the requested partition count and imbalance after the INI load;
4. set the deterministic seed after the INI load when the installed API
   provides `setSeed`;
5. suppress library output;
6. construct and partition the hypergraph;
7. canonicalize the result by sorting nodes within each group and sorting the
   groups, then return only groups containing at least two nodes.

The order is intentional. The official `cut_rKaHyPar_sea20.ini` contains
`seed=-1`, so loading it after `setSeed(0)` would silently undo deterministic
seeding. The explicit runtime values must always be the last values applied.

## Failure and lifecycle semantics

Preserve the current engineering fallback:

- missing KaHyPar import, invalid configuration, constructor incompatibility,
  partition failure, worker timeout, broken pipe, protocol error, or worker
  native exit returns `None`;
- the caller records `degraded_cache` when a prior valid partition cache exists,
  otherwise `degraded_no_cache`;
- no failure may be mislabeled as `success`.

The installed `kahypar==1.3.7` wheel exits its process with status `255` for a
missing INI instead of raising a Python exception. Therefore Python
`try`/`except` in the training process is not a safety boundary. The persistent
worker is the safety boundary: native `exit`, `abort`, or a segmentation fault
can terminate the worker, while the parent detects EOF or a non-live worker,
discards it, and degrades the slot. The next partition attempt may lazily start
a fresh worker; the failed slot is not retried immediately.

The parent terminates and joins a timed-out or failed worker before dropping its
handles. `CleanGraphBuilder.close()` performs an orderly shutdown and join;
training and evaluation entry points call it from `finally`, with an `atexit`
cleanup as a last-resort guard. Normal completion must leave no worker process.

KaHyPar successfully returning blocks that all become singletons after
filtering is not an execution failure. The worker returns an empty list, the
builder marks the update as `success`, and `partition_hyperedges` is empty.
`None` is reserved for execution, configuration, or transport failure and
produces a degraded status.

## Smoke coverage

Extend clean graph smoke coverage with:

- the configured INI path resolves through the repository-root locator and
  exists;
- a small hypergraph produces non-empty partition groups and
  `partition_status=success` when KaHyPar is installed;
- the same input and seed produce the same canonical group-membership set
  across repeated calls; block-number ordering is not part of the assertion;
- an invalid INI in a worker can make the native library exit while the parent
  survives, reports the existing degraded status, and can start a new worker on
  a later request;
- a simulated unavailable KaHyPar path still reports the existing degraded
  status;
- a successful all-singleton result is `success` with zero partition edges,
  while an execution failure is `degraded_*`;
- DAG, k-hop, and attribute hyperedges remain present and unchanged by the
  adapter;
- orderly close, timeout cleanup, and worker-crash cleanup leave no child
  process.

On the installed CPython 3.13 manylinux wheel, the pre-implementation probe
produced one canonical result across 20 repeated calls with seed `0`. This is
evidence for the target environment, not a promise that arbitrary KaHyPar builds
are bit-exact deterministic.

After server sync, run the graph smoke and a real short deterministic checkpoint
evaluation. Acceptance requires at least one KaHyPar `success` slot, non-zero
partition hyperedges, no native configuration error, and no residual worker
process.

## Rollout and experiment gate

Create one independent KaHyPar adapter commit containing the vendored INI,
configuration, worker/client adapter, graph-builder integration, smoke, and
dependency/runbook note. Push `zrj_3`, fast-forward the server, and verify the
installed package version. Do not start the planned `w_c=16` training until the
real short evaluation passes the acceptance criteria.

## Non-goals

- No hyperedge ablation or weighting change.
- No replacement with Mt-KaHyPar or another partitioner.
- No automatic package installation from training code.
- No claim that enabling partition hyperedges alone fixes the RL learning
  problem.
- No per-partition process launch; the worker persists for the builder lifetime
  to avoid thousands of interpreter startups in a normal experiment.
