# Network intuition — a marimo companion app

A self-contained [marimo](https://marimo.io) app that walks through the basics
of network data: reading it, drawing it, measuring it, changing it, and ranking
its nodes by different definitions of *importance*. Built for the
**Network Science Summer School** at Utrecht University (Day 1).

## Live demo

The app is published as a static WebAssembly bundle on GitHub Pages:

**<https://javier.science/marimo_intro_networks/>**

It runs entirely in your browser — no server needed. The bundle is
rebuilt automatically by a GitHub Action on every push to `main`
(see [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)).

## What it does

Five sections, one shared "active network" picked once in the sidebar:

1. **Read it** — the same graph shown as an edge list, an adjacency matrix
   and a GraphML snippet, side by side.
2. **Draw it** — Fruchterman-Reingold, Kamada-Kawai, MDS, and circular layouts;
   colour by detected community or by any node attribute (e.g. football
   conferences, karate factions).
3. **Measure it** — descriptors table + degree / clustering / shortest-path
   distributions.
4. **Change it** — four sliders (rewire %, edges %, clustering bias,
   assortativity bias) that mutate the active network so you can watch each
   metric react.
5. **Rank it** — degree, betweenness, closeness and eigenvector centrality
   in a 2×2 panel with quintile colouring.

Bundled networks (no external data files):

- US college football (Girvan & Newman 2002) — 115 teams, 12 conferences
- Florentine families (Padgett)
- Zachary's karate club
- Krackhardt Kite
- A Watts-Strogatz small-world
- A Barabási-Albert scale-free graph

You can also upload your own edge list (CSV/TSV with `source,target`
columns) or GraphML / GML file.

## Running locally

The app uses [uv](https://docs.astral.sh/uv/) and `python-igraph`. The
helper scripts assume an isolated venv at `~/.uv_envs/day1_network_intuition`
because the original development directory sits on pCloud Drive (which
doesn't support symlinks).

```bash
./run.sh --setup    # first time only — creates the venv and installs deps
./run.sh            # launches marimo edit mode on the app
```

## Rebuilding the WASM bundle locally

```bash
./export_wasm.sh
```

Writes a self-contained static site to `build/` (gitignored) that you
can preview with `python -m http.server`. The CI workflow does the same
thing on push and uploads the output as the Pages artifact.

## License

MIT.
