# Demo

`make demo` walks one injected drift through the whole loop on the real CICIDS2017 stream and
prints each stage as it happens:

1. a stationary slice of Tuesday flows is tiled into a 600,000-event stream (each repetition gets
   fresh event ids and later event times) and replayed at 1,000 events/s;
2. at a known event index a covariate shift (selected features scaled by 2 with noise) begins;
3. the monitor flags consecutive windows and emits a drift event;
4. the orchestrator submits the Prefect run, which registers a challenger and starts shadow mode;
5. after 3,000 labeled shadow events the gate promotes to canary or declines, with its numbers;
6. when a promotion happened, the script rolls back with `make rollback` and reports how long the
   alias took to move.

Run it with the stack up and bootstrapped, and watch the Grafana dashboards (Pipeline health,
Drift monitor, Model lifecycle) at http://localhost:3000.

## Recording

Record a terminal window running `make demo` next to the Model lifecycle dashboard. Export a GIF or
a short video to `docs/demo.gif` (or `docs/demo.mp4`) and embed it in the README under the
quickstart. The recording is a manual step because the stack must be running on real data.
