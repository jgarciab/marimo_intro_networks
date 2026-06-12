#!/usr/bin/env python
"""Headless verification of app.py — run with `uv run python verify_app.py`.

Checks that matter for the WASM (Pyodide) deployment:

1. app.py parses and contains no call to Graph.are_adjacent(), which only
   exists in python-igraph >= 0.11. Pyodide ships its own (often older)
   python-igraph, so any 0.11+ API crashes in the browser even when the
   local venv is fine.
2. For every bundled network, the section-4 "modify" cell runs with the
   clustering-bias and assortativity-bias sliders at both extremes
   (+/-100) without raising, and the degree-preserving swap passes leave
   the edge count AND the degree sequence unchanged.
"""

import ast
import sys
import types


def check_no_new_igraph_api():
    src = open("app.py").read()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("are_adjacent",):
                bad.append(node.lineno)
    assert not bad, (
        f"are_adjacent() called on lines {bad} — needs igraph >= 0.11, "
        "not available in Pyodide's igraph. Use neighbor sets or "
        "get_eid(..., error=False) != -1 instead."
    )
    print("OK  no igraph >= 0.11 API calls (are_adjacent)")


def _stub(value):
    """Mimic a marimo widget: an object with a .value attribute."""
    return types.SimpleNamespace(value=value)


def check_swap_passes():
    import igraph as ig
    import random as rnd_mod

    import app as app_module

    # Build the bundled catalogue through the app's own cell.
    _, defs = app_module.network_catalogue.run(ig=ig)
    bundled = defs["BUNDLED"]

    failures = []
    for name, builder in bundled.items():
        g = builder()
        g.simplify(multiple=True, loops=True)
        deg0 = sorted(g.degree())
        m0 = g.ecount()
        for cb, ab in [(100, 0), (-100, 0), (0, 100), (0, -100), (100, -100)]:
            try:
                _, mdefs = app_module.s4_modify.run(
                    g=g,
                    rnd_mod=rnd_mod,
                    rewire_pct=_stub(0),
                    edges_pct=_stub(100),
                    clustering_bias=_stub(cb),
                    assort_bias=_stub(ab),
                )
                g_mod = mdefs["g_mod"]
            except Exception as e:  # noqa: BLE001
                failures.append(f"{name} cb={cb} ab={ab}: raised {e!r}")
                continue
            if g_mod.ecount() != m0:
                failures.append(
                    f"{name} cb={cb} ab={ab}: edge count {m0} -> "
                    f"{g_mod.ecount()}"
                )
            if sorted(g_mod.degree()) != deg0:
                failures.append(
                    f"{name} cb={cb} ab={ab}: degree sequence changed"
                )
        print(f"OK  swap passes on {name} (m={m0})")

    assert not failures, "\n".join(failures)


def main():
    check_no_new_igraph_api()
    check_swap_passes()
    print("\nAll checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED:\n{e}", file=sys.stderr)
        sys.exit(1)
