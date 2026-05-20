import marimo

__generated_with = "0.23.6"
app = marimo.App(
    width="medium",
    app_title="Network intuition",
)


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import igraph as ig
    import io
    import random as rnd_mod

    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#cccccc",
    })

    # Categorical palette for community / attribute colouring (up to 12 classes).
    PALETTE = (
        "#0b789d", "#e07b00", "#7a9e3b", "#b04a6f", "#5e548e", "#c9a227",
        "#3aa6a0", "#9a4f86", "#d1495b", "#3a7d44", "#2b6cb0", "#8c6b3f",
    )
    ACCENT = "#0b789d"
    NEUTRAL_NODE = "#0b789d"
    # Light grey used for edges in every section. NOTE: igraph's matplotlib
    # backend silently drops the alpha channel of an RGBA tuple, so use a
    # hex string. Slightly darker than day3b's "#bbbbbb" for a bit more
    # contrast on the white background.
    EDGE_COLOR = "#aaaaaa"

    # Hard cap for WASM (Pyodide). Beyond ~500 nodes the browser slows on
    # every slider tick because matplotlib + ig.plot is the dominant cost.
    MAX_NODES = 500
    return ACCENT, EDGE_COLOR, MAX_NODES, NEUTRAL_NODE, PALETTE, ig, io, mo, np, pd, plt, rnd_mod


# -----------------------------------------------------------------------------
# Inlined network catalogue
# -----------------------------------------------------------------------------


@app.cell
def network_catalogue(ig):
    # Florentine families (Padgett 1994) — built from the canonical edge list.
    _flor_names = (
        "ACCIAIUOL", "ALBIZZI", "BARBADORI", "BISCHERI", "CASTELLAN",
        "GINORI", "GUADAGNI", "LAMBERTES", "MEDICI", "PAZZI", "PERUZZI",
        "PUCCI", "RIDOLFI", "SALVIATI", "STROZZI", "TORNABUON",
    )
    _flor_edges = (
        (0, 8), (1, 5), (1, 6), (1, 8), (2, 4), (2, 5), (2, 8), (2, 10),
        (3, 6), (3, 7), (3, 10), (3, 14), (4, 7), (4, 10), (4, 14),
        (5, 8), (6, 7), (6, 15), (7, 10), (8, 9), (8, 12), (8, 13),
        (8, 15), (9, 13), (10, 14), (12, 14), (12, 15),
    )

    # Zachary karate factions (Zachary 1977).
    _zachary_factions = (
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    )

    # US college football network (Girvan & Newman 2002) — 115 teams,
    # 613 games, 12 conferences. Inlined so the app needs no data files.
    FOOTBALL_NAMES = (
        "BrighamYoung", "FloridaState", "Iowa", "KansasState", "NewMexico",
        "TexasTech", "PennState", "SouthernCalifornia", "ArizonaState",
        "SanDiegoState", "Baylor", "NorthTexas", "NorthernIllinois",
        "Northwestern", "WesternMichigan", "Wisconsin", "Wyoming", "Auburn",
        "Akron", "VirginiaTech", "Alabama", "UCLA", "Arizona", "Utah",
        "ArkansasState", "NorthCarolinaState", "BallState", "Florida",
        "BoiseState", "BostonCollege", "WestVirginia", "BowlingGreenState",
        "Michigan", "Virginia", "Buffalo", "Syracuse", "CentralFlorida",
        "GeorgiaTech", "CentralMichigan", "Purdue", "Colorado",
        "ColoradoState", "Connecticut", "EasternMichigan", "EastCarolina",
        "Duke", "FresnoState", "OhioState", "Houston", "Rice", "Idaho",
        "Washington", "Kansas", "SouthernMethodist", "Kent", "Pittsburgh",
        "Kentucky", "Louisville", "LouisianaTech", "LouisianaMonroe",
        "Minnesota", "MiamiOhio", "Vanderbilt", "MiddleTennesseeState",
        "Illinois", "MississippiState", "Memphis", "Nevada", "Oregon",
        "NewMexicoState", "SouthCarolina", "Ohio", "IowaState",
        "SanJoseState", "Nebraska", "SouthernMississippi", "Tennessee",
        "Stanford", "WashingtonState", "Temple", "Navy", "TexasA&M",
        "NotreDame", "TexasElPaso", "Oklahoma", "Toledo", "Tulane",
        "Mississippi", "Tulsa", "NorthCarolina", "UtahState", "Army",
        "Cincinnati", "AirForce", "Rutgers", "Georgia", "LouisianaState",
        "LouisianaLafayette", "Texas", "Marshall", "MichiganState",
        "MiamiFlorida", "Missouri", "Clemson", "NevadaLasVegas",
        "WakeForest", "Indiana", "OklahomaState", "OregonState", "Maryland",
        "TexasChristian", "California", "AlabamaBirmingham", "Arkansas",
        "Hawaii",
    )
    FOOTBALL_CONF = (
        7, 0, 2, 3, 7, 3, 2, 8, 8, 7, 3, 10, 6, 2, 6, 2, 7, 9, 6, 1,
        9, 8, 8, 7, 10, 0, 6, 9, 11, 1, 1, 6, 2, 0, 6, 1, 5, 0, 6, 2,
        3, 7, 5, 6, 4, 0, 11, 2, 4, 11, 10, 8, 3, 11, 6, 1, 9, 4, 11, 10,
        2, 6, 9, 10, 2, 9, 4, 11, 8, 10, 9, 6, 3, 11, 3, 4, 9, 8, 8, 1,
        5, 3, 5, 11, 3, 6, 4, 9, 11, 0, 5, 4, 4, 7, 1, 9, 9, 10, 3, 6,
        2, 1, 3, 0, 7, 0, 2, 3, 8, 0, 4, 8, 4, 9, 11,
    )
    CONFERENCE_NAMES = (
        "Atlantic Coast", "Big East", "Big Ten", "Big Twelve",
        "Conference USA", "Independents", "Mid-American", "Mountain West",
        "Pacific Ten", "Southeastern", "Sun Belt", "Western Athletic",
    )
    FOOTBALL_EDGES = (
        (0, 1), (0, 4), (0, 9), (0, 16), (0, 23), (0, 33), (0, 35), (0, 41),
        (0, 65), (0, 90), (0, 93), (0, 104), (1, 25), (1, 27), (1, 33),
        (1, 37), (1, 45), (1, 57), (1, 89), (1, 101), (1, 103), (1, 105),
        (1, 109), (2, 3), (2, 6), (2, 13), (2, 14), (2, 15), (2, 47),
        (2, 60), (2, 64), (2, 72), (2, 74), (2, 100), (2, 106), (3, 5),
        (3, 11), (3, 26), (3, 40), (3, 52), (3, 58), (3, 72), (3, 74),
        (3, 81), (3, 84), (3, 102), (4, 5), (4, 9), (4, 16), (4, 23),
        (4, 28), (4, 41), (4, 69), (4, 93), (4, 104), (4, 108), (5, 10),
        (5, 11), (5, 52), (5, 74), (5, 81), (5, 84), (5, 90), (5, 97),
        (5, 98), (5, 107), (6, 7), (6, 32), (6, 39), (6, 47), (6, 55),
        (6, 58), (6, 60), (6, 64), (6, 85), (6, 100), (6, 106), (7, 8),
        (7, 21), (7, 22), (7, 40), (7, 68), (7, 73), (7, 77), (7, 78),
        (7, 82), (7, 108), (7, 111), (8, 9), (8, 21), (8, 22), (8, 41),
        (8, 51), (8, 68), (8, 77), (8, 78), (8, 90), (8, 111), (9, 16),
        (9, 22), (9, 23), (9, 41), (9, 64), (9, 93), (9, 104), (9, 108),
        (10, 11), (10, 60), (10, 72), (10, 74), (10, 81), (10, 84),
        (10, 98), (10, 102), (10, 107), (11, 24), (11, 28), (11, 50),
        (11, 69), (11, 90), (11, 97), (11, 104), (12, 13), (12, 14),
        (12, 17), (12, 18), (12, 26), (12, 34), (12, 36), (12, 38),
        (12, 43), (12, 85), (13, 15), (13, 32), (13, 39), (13, 45),
        (13, 60), (13, 64), (13, 100), (13, 106), (13, 110), (14, 15),
        (14, 26), (14, 38), (14, 43), (14, 54), (14, 71), (14, 85),
        (14, 99), (15, 32), (15, 39), (15, 47), (15, 60), (15, 68),
        (15, 92), (15, 100), (15, 106), (15, 114), (16, 17), (16, 23),
        (16, 38), (16, 41), (16, 67), (16, 81), (16, 93), (16, 104),
        (17, 20), (17, 27), (17, 58), (17, 62), (17, 65), (17, 87),
        (17, 95), (17, 96), (17, 113), (18, 19), (18, 31), (18, 34),
        (18, 36), (18, 38), (18, 42), (18, 54), (18, 61), (18, 71),
        (18, 99), (19, 29), (19, 30), (19, 33), (19, 35), (19, 36),
        (19, 44), (19, 55), (19, 79), (19, 94), (19, 101), (20, 21),
        (20, 36), (20, 62), (20, 65), (20, 70), (20, 75), (20, 76),
        (20, 87), (20, 96), (20, 113), (21, 22), (21, 32), (21, 46),
        (21, 51), (21, 68), (21, 77), (21, 108), (21, 111), (22, 23),
        (22, 47), (22, 51), (22, 68), (22, 77), (22, 78), (22, 108),
        (23, 41), (23, 78), (23, 90), (23, 93), (23, 104), (23, 111),
        (24, 25), (24, 28), (24, 50), (24, 66), (24, 69), (24, 84),
        (24, 87), (24, 90), (24, 110), (25, 33), (25, 37), (25, 45),
        (25, 53), (25, 89), (25, 103), (25, 105), (25, 106), (25, 109),
        (26, 27), (26, 34), (26, 38), (26, 42), (26, 43), (26, 61),
        (26, 85), (27, 56), (27, 62), (27, 63), (27, 65), (27, 70),
        (27, 76), (27, 95), (27, 96), (28, 38), (28, 50), (28, 69),
        (28, 78), (28, 90), (28, 113), (29, 30), (29, 35), (29, 42),
        (29, 55), (29, 79), (29, 80), (29, 82), (29, 91), (29, 94),
        (29, 101), (30, 35), (30, 44), (30, 50), (30, 55), (30, 79),
        (30, 82), (30, 94), (30, 101), (30, 109), (31, 32), (31, 34),
        (31, 43), (31, 54), (31, 55), (31, 61), (31, 71), (31, 79),
        (31, 85), (31, 99), (32, 39), (32, 47), (32, 49), (32, 64),
        (32, 100), (32, 106), (33, 37), (33, 45), (33, 89), (33, 103),
        (33, 105), (33, 109), (34, 35), (34, 42), (34, 54), (34, 61),
        (34, 71), (34, 94), (34, 99), (35, 44), (35, 55), (35, 79),
        (35, 92), (35, 94), (35, 101), (36, 37), (36, 43), (36, 58),
        (36, 59), (37, 45), (37, 80), (37, 89), (37, 95), (37, 103),
        (37, 105), (37, 109), (38, 39), (38, 43), (38, 54), (38, 71),
        (38, 85), (39, 47), (39, 54), (39, 60), (39, 82), (39, 100),
        (39, 106), (40, 41), (40, 51), (40, 52), (40, 72), (40, 74),
        (40, 81), (40, 98), (40, 102), (40, 107), (41, 67), (41, 93),
        (41, 104), (42, 43), (42, 57), (42, 63), (43, 61), (43, 70),
        (43, 79), (43, 85), (44, 45), (44, 48), (44, 57), (44, 66),
        (44, 75), (44, 86), (44, 91), (44, 112), (45, 62), (45, 89),
        (45, 103), (45, 105), (45, 109), (46, 47), (46, 49), (46, 53),
        (46, 67), (46, 73), (46, 83), (46, 88), (46, 110), (46, 111),
        (46, 114), (47, 60), (47, 61), (47, 64), (47, 100), (48, 49),
        (48, 53), (48, 57), (48, 66), (48, 75), (48, 86), (48, 91),
        (48, 92), (48, 96), (48, 98), (49, 53), (49, 67), (49, 73),
        (49, 83), (49, 84), (49, 88), (49, 110), (49, 114), (50, 51),
        (50, 68), (50, 69), (50, 78), (50, 90), (51, 68), (51, 77),
        (51, 78), (51, 101), (51, 108), (51, 111), (52, 53), (52, 72),
        (52, 74), (52, 84), (52, 98), (52, 102), (52, 112), (53, 67),
        (53, 73), (53, 83), (53, 86), (53, 88), (53, 110), (53, 114),
        (54, 55), (54, 61), (54, 71), (54, 99), (55, 79), (55, 89),
        (55, 94), (55, 101), (56, 57), (56, 62), (56, 65), (56, 70),
        (56, 76), (56, 87), (56, 95), (56, 96), (56, 106), (57, 75),
        (57, 86), (57, 91), (57, 92), (57, 112), (58, 59), (58, 63),
        (58, 88), (58, 97), (58, 101), (58, 114), (59, 60), (59, 63),
        (59, 66), (59, 76), (59, 97), (59, 113), (60, 64), (60, 71),
        (60, 106), (61, 62), (61, 71), (61, 92), (61, 99), (62, 70),
        (62, 76), (62, 87), (62, 95), (62, 105), (63, 64), (63, 65),
        (63, 97), (63, 109), (63, 112), (64, 100), (64, 106), (64, 111),
        (65, 66), (65, 70), (65, 87), (65, 96), (65, 113), (66, 75),
        (66, 76), (66, 86), (66, 91), (66, 92), (66, 112), (67, 68),
        (67, 73), (67, 83), (67, 88), (67, 104), (67, 110), (67, 114),
        (68, 78), (68, 108), (68, 111), (69, 70), (69, 83), (69, 88),
        (69, 90), (69, 91), (69, 95), (70, 76), (70, 95), (70, 103),
        (70, 113), (71, 72), (71, 99), (72, 74), (72, 81), (72, 102),
        (72, 104), (72, 107), (73, 74), (73, 77), (73, 83), (73, 88),
        (73, 110), (73, 114), (74, 82), (74, 84), (74, 102), (75, 76),
        (75, 86), (75, 92), (75, 107), (75, 112), (76, 95), (76, 96),
        (76, 113), (77, 78), (77, 82), (77, 98), (77, 108), (77, 111),
        (78, 108), (78, 111), (79, 80), (79, 94), (79, 101), (79, 109),
        (80, 82), (80, 85), (80, 86), (80, 91), (80, 93), (80, 94),
        (80, 105), (80, 110), (81, 82), (81, 83), (81, 84), (81, 98),
        (81, 107), (82, 93), (82, 94), (82, 100), (83, 84), (83, 88),
        (83, 110), (83, 114), (84, 98), (84, 107), (85, 99), (86, 87),
        (86, 91), (86, 92), (86, 97), (87, 95), (87, 96), (87, 104),
        (87, 113), (88, 89), (88, 107), (88, 110), (88, 114), (89, 99),
        (89, 103), (89, 105), (89, 109), (91, 92), (91, 93), (91, 112),
        (92, 106), (92, 112), (93, 104), (94, 101), (95, 113), (96, 112),
        (96, 113), (97, 98), (97, 112), (98, 102), (98, 107), (99, 100),
        (100, 102), (102, 103), (102, 107), (103, 105), (103, 109),
        (104, 114), (105, 109), (108, 111), (110, 114),
    )

    def _build(names, edges, attr=None, attr_name=None, value_names=None):
        g_ = ig.Graph(n=len(names), edges=list(edges), directed=False)
        g_.vs["name"] = list(names)
        if attr is not None:
            g_.vs[attr_name] = list(attr)
            g_["_attr_name"] = attr_name
            g_["_attr_value_names"] = list(value_names) if value_names else None
        return g_

    def _football():
        return _build(
            FOOTBALL_NAMES, FOOTBALL_EDGES,
            attr=FOOTBALL_CONF, attr_name="Conference",
            value_names=CONFERENCE_NAMES,
        )

    def _karate():
        g_ = ig.Graph.Famous("Zachary")
        g_.vs["name"] = [str(i) for i in range(g_.vcount())]
        g_.vs["Faction"] = list(_zachary_factions)
        g_["_attr_name"] = "Faction"
        g_["_attr_value_names"] = ["Mr Hi", "Officer"]
        return g_

    def _florentine():
        return _build(_flor_names, _flor_edges)

    def _kite():
        g_ = ig.Graph.Famous("Krackhardt_Kite")
        g_.vs["name"] = [str(i) for i in range(g_.vcount())]
        return g_

    def _smallworld():
        g_ = ig.Graph.Watts_Strogatz(dim=1, size=40, nei=3, p=0.1)
        g_.vs["name"] = [f"n{i}" for i in range(g_.vcount())]
        return g_

    def _scalefree():
        g_ = ig.Graph.Barabasi(n=80, m=2, directed=False)
        g_.vs["name"] = [f"n{i}" for i in range(g_.vcount())]
        return g_

    BUNDLED = {
        "Football (115 teams, 12 conferences)": _football,
        "Florentine families (16 nodes)": _florentine,
        "Karate club (34 nodes, 2 factions)": _karate,
        "Krackhardt Kite (10 nodes)": _kite,
        "Small-world (40 nodes)": _smallworld,
        "Scale-free (80 nodes)": _scalefree,
    }
    return (BUNDLED,)


# -----------------------------------------------------------------------------
# Sidebar: network chooser (visible from every section)
# -----------------------------------------------------------------------------


@app.cell
def chooser_widgets(BUNDLED, mo):
    bundled_choice = mo.ui.dropdown(
        options=list(BUNDLED.keys()),
        value="Football (115 teams, 12 conferences)",
        label="Bundled network",
    )
    file_upload = mo.ui.file(
        kind="button",
        filetypes=[".csv", ".tsv", ".txt", ".graphml", ".gml", ".xml"],
        label="Upload edge list or GraphML/GML",
        multiple=False,
    )
    return bundled_choice, file_upload


@app.cell
def build_active_graph(BUNDLED, MAX_NODES, bundled_choice, file_upload, ig, io, pd):
    import os
    import tempfile

    def _read_uploaded(file_obj):
        """Parse an uploaded file (CSV/TSV edge list, GraphML, or GML)."""
        name = file_obj.name
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        raw = file_obj.contents
        if isinstance(raw, str):
            raw_bytes = raw.encode("utf-8")
        else:
            raw_bytes = raw

        if ext in ("graphml", "xml"):
            # igraph's reader needs a filename — write to /tmp (works in
            # Pyodide too).
            with tempfile.NamedTemporaryFile(
                suffix=".graphml", delete=False
            ) as tf:
                tf.write(raw_bytes)
                tf_path = tf.name
            try:
                return ig.Graph.Read_GraphML(tf_path)
            finally:
                try:
                    os.unlink(tf_path)
                except OSError:
                    pass

        if ext == "gml":
            with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tf:
                tf.write(raw_bytes)
                tf_path = tf.name
            try:
                return ig.Graph.Read_GML(tf_path)
            finally:
                try:
                    os.unlink(tf_path)
                except OSError:
                    pass

        # Otherwise: edge list (CSV/TSV/TXT)
        text = raw_bytes.decode("utf-8", errors="replace")
        sep = "\t" if text.count("\t") > text.count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
        df.columns = [c.lower().strip() for c in df.columns]
        if "source" not in df.columns or "target" not in df.columns:
            raise ValueError(
                "Edge list must have columns named 'source' and 'target'."
            )
        edges = list(zip(df["source"].astype(str), df["target"].astype(str)))
        return ig.Graph.TupleList(edges, directed=False)

    def _detect_attrs(graph):
        """Return a list of vertex attribute names that look categorical."""
        skip = {"name", "id", "x", "y", "_pos", "label", "nodelabel"}
        out = []
        for a in graph.vs.attributes():
            if a.lower() in skip or a.startswith("_"):
                continue
            try:
                vals = graph.vs[a]
                unique = set(vals)
            except TypeError:
                continue
            # Categorical: few enough unique values to colour by
            cap = max(20, graph.vcount() // 4)
            if 1 < len(unique) <= cap:
                out.append(a)
        return out

    upload_warning = None
    source_label = ""
    g_active = None

    if file_upload.value and len(file_upload.value) > 0:
        _f = file_upload.value[0]
        try:
            _g = _read_uploaded(_f)
            _g.simplify(multiple=True, loops=True)
            if _g.vcount() > MAX_NODES:
                _g = _g.connected_components().giant()
                if _g.vcount() > MAX_NODES:
                    _keep = sorted(
                        range(_g.vcount()), key=lambda i: -_g.degree(i)
                    )[:MAX_NODES]
                    _g = _g.subgraph(_keep)
                upload_warning = (
                    f"Upload had more than {MAX_NODES} nodes; trimmed to "
                    f"the largest component ({_g.vcount()} nodes)."
                )
            g_active = _g
            source_label = f"upload: {_f.name}"
        except Exception as _e:
            upload_warning = f"Upload error: {_e}. Falling back to bundled."
            g_active = None

    if g_active is None:
        g_active = BUNDLED[bundled_choice.value]()
        source_label = bundled_choice.value

    g_active.simplify(multiple=True, loops=True)
    # Make sure every vertex has a usable label
    if "name" not in g_active.vs.attributes() or any(
        v["name"] is None for v in g_active.vs
    ):
        if "id" in g_active.vs.attributes():
            g_active.vs["name"] = [
                str(v) if v is not None else str(i)
                for i, v in enumerate(g_active.vs["id"])
            ]
        else:
            g_active.vs["name"] = [str(i) for i in range(g_active.vcount())]

    g = g_active
    # Attributes available for colouring — bundled networks declare a
    # primary one via g["_attr_name"]; for uploads we sniff them out.
    primary_attr = g["_attr_name"] if "_attr_name" in g.attributes() else None
    attr_value_names = (
        g["_attr_value_names"] if "_attr_value_names" in g.attributes() else None
    )
    detected = _detect_attrs(g)
    if primary_attr is not None and primary_attr in detected:
        # put the bundled "primary" attribute first
        detected = [primary_attr] + [a for a in detected if a != primary_attr]
    attr_names = detected
    attr_name = attr_names[0] if attr_names else None
    return attr_name, attr_names, attr_value_names, g, source_label, upload_warning


@app.cell
def sidebar_cell(bundled_choice, file_upload, mo, source_label, upload_warning):
    _items = [
        mo.md("### Active network"),
        bundled_choice,
        mo.md("_Or upload your own:_"),
        file_upload,
        mo.md(f"**Currently:**  \n{source_label}"),
    ]
    if upload_warning is not None:
        _items.append(mo.md(f"> ⚠️ {upload_warning}"))
    _items.append(mo.md("---"))
    _items.append(mo.md(
        "_The active network drives every section below — pick it once "
        "here, then scroll._"
    ))
    mo.sidebar(_items)
    return


@app.cell
def title(mo):
    mo.md(r"""
    # From data to a story about a network

    Pick a network in the sidebar — bundled, your own edge list, or your
    own GraphML — and we'll do five things with that same network:

    1. **Read it** — three data formats, same graph
    2. **Draw it** — layouts, colours, and visual encoding
    3. **Measure it** — averages and the distributions behind them
    4. **Change it** — rewire and resize, watch the metrics react
    5. **Rank it** — four definitions of *important*, four answers

    The default is the **US college football network** (Girvan & Newman
    2002): 115 teams, 613 games, 12 conferences. Small enough to keep on
    screen, large enough to have real community structure.
    """)
    return


# -----------------------------------------------------------------------------
# Section 1 — Three data formats
# -----------------------------------------------------------------------------


@app.cell
def section1_header(mo):
    mo.md(r"""
    ---
    ## 1 · Read it — three data formats

    The same graph can be written down in many ways. The choice depends
    on what you want to keep around and who you are sharing it with. Below,
    the active network in three flavours, side by side.
    """)
    return


@app.cell
def s1_three_views(attr_name, g, mo, np, pd, plt):
    _names = g.vs["name"]
    _n = g.vcount()

    # 1. Edge list (narrow)
    _rows = [(_names[e.source], _names[e.target]) for e in g.es]
    _edgelist_df = pd.DataFrame(_rows, columns=["source", "target"])

    # 2. Adjacency matrix
    _A = np.array(g.get_adjacency().data)
    _fig, _ax = plt.subplots(figsize=(5.2, 4.8))
    _ax.imshow(_A, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    if _n <= 30:
        _ax.set_xticks(range(_n))
        _ax.set_yticks(range(_n))
        _ax.set_xticklabels(_names, rotation=90, fontsize=8)
        _ax.set_yticklabels(_names, fontsize=8)
    else:
        _ax.set_xticks([])
        _ax.set_yticks([])
    _ax.set_title(f"Adjacency matrix ({_n}×{_n})")
    _ax.grid(False)
    plt.tight_layout()

    # 3. GraphML snippet, including the (primary) node attribute when present
    _attr_vals = g.vs[attr_name] if attr_name else None
    _gml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="d0" for="node" attr.name="name" attr.type="string"/>',
    ]
    if attr_name is not None:
        _gml_lines.append(
            f'  <key id="d1" for="node" attr.name="{attr_name}" attr.type="string"/>'
        )
    _gml_lines.append('  <graph edgedefault="undirected">')
    for _i, _nm in enumerate(_names):
        _node = (
            f'    <node id="n{_i}"><data key="d0">{_nm}</data>'
        )
        if attr_name is not None:
            _node += f'<data key="d1">{_attr_vals[_i]}</data>'
        _node += "</node>"
        _gml_lines.append(_node)
    for _e in g.es:
        _gml_lines.append(
            f'    <edge source="n{_e.source}" target="n{_e.target}"/>'
        )
    _gml_lines.append("  </graph>")
    _gml_lines.append("</graphml>")
    _graphml_text = "\n".join(_gml_lines)
    _snippet = (
        "\n".join(_graphml_text.splitlines()[:18])
        + "\n  ...\n</graphml>"
    )

    # Build a GraphML snippet that shows BOTH some nodes and some edges
    # — split into a head (the <key>/first nodes) and a tail (first edges
    # + closing tag), joined by an ellipsis.
    _all_lines = _graphml_text.splitlines()
    # locate where edges begin
    _edge_start = next(
        (i for i, ln in enumerate(_all_lines) if ln.lstrip().startswith("<edge")),
        len(_all_lines),
    )
    _max_nodes = 6
    _max_edges = 8
    _head_end = min(_edge_start, _all_lines.index("  <graph edgedefault=\"undirected\">") + 1 + _max_nodes)
    _head = _all_lines[:_head_end]
    if _head_end < _edge_start:
        _head.append("    ...")
    _tail_lines = _all_lines[_edge_start:_edge_start + _max_edges]
    if _edge_start + _max_edges < len(_all_lines) - 2:
        _tail_lines.append("    ...")
    _tail_lines += _all_lines[-2:]  # </graph></graphml>
    _snippet_both = "\n".join(_head + _tail_lines)

    _col_edgelist = mo.vstack([
        mo.md("**1. Edge list (CSV/TSV)**"),
        mo.ui.table(_edgelist_df, page_size=8, selection=None),
        mo.md(
            f"_{len(_rows)} edges._ Two columns, one row per relationship. "
            "Easy to share, easy to read. No place for node attributes."
        ),
    ])
    _col_adj = mo.vstack([
        mo.md("**2. Adjacency matrix**"),
        _fig,
        mo.md(
            "One cell per ordered pair. Linear algebra is easy on this "
            "shape (paths = matrix powers, communities = eigenvectors), "
            "but memory is _O(n²)_ — sparse storage matters for large "
            "graphs."
        ),
    ])

    _row_top = mo.hstack(
        [_col_edgelist, _col_adj],
        widths=[1.0, 1.2],
        gap=1.5,
        align="start",
    )
    _row_bottom = mo.vstack([
        mo.md("**3. GraphML / GML (interchange XML)** — nodes _and_ edges in one file"),
        mo.md(f"```xml\n{_snippet_both}\n```"),
        mo.md(
            "Keeps **node properties** (here `name`"
            + (f", `{attr_name}`" if attr_name else "")
            + ") _and_ edge attributes (weight, type, timestamp). "
            "Supports directed, undirected, mixed, weighted and "
            "multi-graphs. Verbose but lossless — the right format "
            "when the data is more than a bare list of pairs. "
            "**Rule of thumb:** GraphML in general, CSV for small / "
            "simple cases (you usually need two tables — one for "
            "edges, one for node attributes), JSON for the web."
        ),
    ])
    mo.vstack([_row_top, _row_bottom], gap=1.5)
    return


# -----------------------------------------------------------------------------
# Section 2 — Visualization (controls on the left, plot on the right)
# -----------------------------------------------------------------------------


@app.cell
def section2_header(mo):
    mo.md(r"""
    ---
    ## 2 · Draw it — layouts and visual encoding

    A drawing is a *choice*, not a fact. The same network can look like a
    clear hierarchy under one layout and a chaotic blob under another.
    Force-directed layouts (Fruchterman-Reingold, Kamada-Kawai) let
    communities emerge as visual clusters; circular and random layouts
    deliberately don't. Pick one, choose what to colour by, and notice
    which structural facts survive across layouts.

    Node positions are computed **once** per layout — changing node
    size or colouring does not reshuffle the nodes.
    """)
    return


@app.cell
def s2_widgets(attr_names, g, mo):
    _color_options = ["None (single colour)"]
    for _a in attr_names:
        _color_options.append(f"Attribute · {_a}")
    _color_options.append("Detected community")
    _default_color = (
        f"Attribute · {attr_names[0]}" if attr_names else "Detected community"
    )

    layout_choice = mo.ui.dropdown(
        options=["Fruchterman-Reingold", "Kamada-Kawai", "MDS", "Circle"],
        value="Fruchterman-Reingold",
        label="Layout",
    )
    color_choice = mo.ui.dropdown(
        options=_color_options,
        value=_default_color,
        label="Colour by",
    )
    node_size = mo.ui.slider(
        start=10, stop=80, step=2, value=22, label="Node size",
    )
    show_labels = mo.ui.checkbox(
        value=(g.vcount() <= 40), label="Show node labels",
    )
    return color_choice, layout_choice, node_size, show_labels


@app.cell
def s2_layout_cache(g, layout_choice):
    # Depends only on g and layout_choice — node-size / alpha changes
    # do NOT trigger a recompute.
    _layout_map = {
        "Fruchterman-Reingold": lambda: g.layout_fruchterman_reingold(niter=500),
        "Kamada-Kawai": g.layout_kamada_kawai,
        "MDS": g.layout_mds,
        "Circle": g.layout_circle,
    }
    layout_coords = [tuple(row) for row in _layout_map[layout_choice.value]().coords]
    try:
        community_membership = g.community_multilevel().membership
    except Exception:
        community_membership = [0] * g.vcount()
    return community_membership, layout_coords


@app.cell
def s2_plot(
    EDGE_COLOR,
    NEUTRAL_NODE,
    PALETTE,
    attr_name,
    attr_value_names,
    color_choice,
    community_membership,
    g,
    ig,
    layout_choice,
    layout_coords,
    mo,
    node_size,
    plt,
    show_labels,
):
    _choice = color_choice.value
    _legend_pairs = None  # list of (label, color) for legend
    _legend_title = None

    if _choice == "None (single colour)":
        _colors = [NEUTRAL_NODE] * g.vcount()
    elif _choice.startswith("Attribute · "):
        _active_attr = _choice[len("Attribute · "):]
        _vals_raw = g.vs[_active_attr]
        # Map raw values (which can be int, str, anything) to colour indices
        # so we can colour categorical attributes that aren't integers.
        _unique = sorted(set(_vals_raw), key=lambda x: str(x))
        _idx = {v: i for i, v in enumerate(_unique)}
        _colors = [PALETTE[_idx[v] % len(PALETTE)] for v in _vals_raw]
        # Use bundled value names only when the attribute matches the primary.
        _use_names = (
            attr_value_names is not None and _active_attr == attr_name
        )
        _legend_pairs = []
        for _v in _unique:
            if _use_names and isinstance(_v, int) and 0 <= _v < len(attr_value_names):
                _label = attr_value_names[_v]
            else:
                _label = str(_v)
            _legend_pairs.append((_label, PALETTE[_idx[_v] % len(PALETTE)]))
        _legend_title = _active_attr
    else:  # Detected community
        _colors = [
            PALETTE[m % len(PALETTE)] for m in community_membership
        ]

    _fig, _ax = plt.subplots(figsize=(7.5, 6.5))
    _ax.set_facecolor("white")
    _ax.grid(False)
    _ax.set_xticks([])
    _ax.set_yticks([])
    for _side in ("top", "right", "bottom", "left"):
        _ax.spines[_side].set_visible(False)

    _vlabels = g.vs["name"] if show_labels.value else [""] * g.vcount()
    ig.plot(
        g,
        target=_ax,
        layout=layout_coords,
        vertex_color=_colors,
        vertex_size=node_size.value,
        vertex_frame_width=0,
        vertex_label=_vlabels,
        vertex_label_size=9,
        vertex_label_color="#222222",
        edge_color=EDGE_COLOR,
        edge_width=1.0,
    )
    if _legend_pairs is not None and len(_legend_pairs) <= 14:
        _handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=_c, markeredgecolor="#333", markersize=8,
                label=_lbl,
            )
            for _lbl, _c in _legend_pairs
        ]
        _ax.legend(
            handles=_handles, title=_legend_title, loc="upper right",
            frameon=True, fontsize=8, title_fontsize=9, ncol=1,
        )
    _ax.set_title(
        f"Layout: {layout_choice.value}  ·  n={g.vcount()}, m={g.ecount()}"
    )
    plt.tight_layout()

    _controls = mo.vstack([
        layout_choice, color_choice, node_size, show_labels,
    ], gap=0.8)
    mo.hstack(
        [_controls, _fig],
        widths=[0.9, 2.4],
        gap=1.5,
        align="start",
    )
    return


# -----------------------------------------------------------------------------
# Section 3 — Averages and distributions (one row)
# -----------------------------------------------------------------------------


@app.cell
def section3_header(mo):
    mo.md(r"""
    ---
    ## 3 · Measure it — averages and distributions

    Averages compress a lot of structure into a single number. Two networks
    can share the same mean degree and look nothing alike — the **spread**
    is where the story usually lives. The table on the left is the
    snapshot; the histograms on the right are the spread (red dashed line
    = the mean, the number that appears in the table).
    """)
    return


@app.cell
def s3_combined(ACCENT, g, mo, np, plt):
    # --- left: stats table ---
    _n = g.vcount()
    _m = g.ecount()
    _comps = g.connected_components()
    _ncomp = len(_comps)
    _density = g.density()
    _avg_deg = 2 * _m / _n if _n else 0.0
    _trans = g.transitivity_undirected(mode="zero")
    _avg_local = float(np.mean(g.transitivity_local_undirected(mode="zero")))
    try:
        _assort = g.assortativity_degree(directed=False)
        if _assort is None:
            _assort = float("nan")
    except Exception:
        _assort = float("nan")
    if g.is_connected():
        _diam = g.diameter()
        _avg_path = g.average_path_length()
        _diam_note = ""
    else:
        _gc = _comps.giant()
        _diam = _gc.diameter()
        _avg_path = _gc.average_path_length()
        _diam_note = f" *(largest comp.)*"

    def _fmt(x):
        if isinstance(x, float):
            if np.isnan(x):
                return "—"
            return f"{x:.3f}"
        return str(x)

    _table = (
        "| Descriptor | Value |\n"
        "|---|---|\n"
        f"| Nodes (n) | {_n} |\n"
        f"| Edges (m) | {_m} |\n"
        f"| Avg. degree ⟨k⟩ | {_fmt(_avg_deg)} |\n"
        f"| Density | {_fmt(_density)} |\n"
        f"| Components | {_ncomp} |\n"
        f"| Diameter | {_fmt(_diam)}{_diam_note} |\n"
        f"| Avg. path | {_fmt(_avg_path)}{_diam_note} |\n"
        f"| Global clustering | {_fmt(_trans)} |\n"
        f"| Avg. local clustering | {_fmt(_avg_local)} |\n"
        f"| Degree assortativity | {_fmt(_assort)} |\n"
    )

    # --- right: three distribution plots ---
    _degrees = np.array(g.degree())
    _clust = np.array(g.transitivity_local_undirected(mode="zero"))
    _D = np.array(g.distances())
    _D_flat = _D[np.isfinite(_D)]
    _D_flat = _D_flat[_D_flat > 0]

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.2))
    _axes[0].hist(
        _degrees,
        bins=max(5, min(25, len(set(_degrees)))),
        color=ACCENT, edgecolor="white",
    )
    _axes[0].axvline(_degrees.mean(), color="#c0223b", linestyle="--",
                     linewidth=1.1, label=f"⟨k⟩={_degrees.mean():.2f}")
    _axes[0].set_title("Degree", fontsize=11)
    _axes[0].set_xlabel("k")
    _axes[0].legend(frameon=False, fontsize=8)

    _axes[1].hist(_clust, bins=15, color=ACCENT, edgecolor="white")
    _axes[1].axvline(_clust.mean(), color="#c0223b", linestyle="--",
                     linewidth=1.1, label=f"⟨C⟩={_clust.mean():.2f}")
    _axes[1].set_title("Local clustering", fontsize=11)
    _axes[1].set_xlabel("C_i")
    _axes[1].legend(frameon=False, fontsize=8)

    if len(_D_flat) > 0:
        _bins = np.arange(_D_flat.min(), _D_flat.max() + 2) - 0.5
        _axes[2].hist(_D_flat, bins=_bins, color=ACCENT, edgecolor="white")
        _axes[2].axvline(_D_flat.mean(), color="#c0223b", linestyle="--",
                         linewidth=1.1, label=f"⟨d⟩={_D_flat.mean():.2f}")
        _axes[2].legend(frameon=False, fontsize=8)
    _axes[2].set_title("Shortest paths", fontsize=11)
    _axes[2].set_xlabel("d(i,j)")

    for _ax in _axes:
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    plt.tight_layout()

    mo.hstack(
        [mo.md(_table), _fig],
        widths=[0.9, 2.6],
        gap=1.5,
        align="start",
    )
    return


# -----------------------------------------------------------------------------
# Section 4 — Watch the metrics react
# -----------------------------------------------------------------------------


@app.cell
def section4_header(mo):
    mo.md(r"""
    ---
    ## 4 · Change it — watch the metrics react

    The best way to understand a metric is to break it. Start from the
    active network and modify it. The original metrics stay on the left;
    the modified ones update on the right as you move the sliders.

    Things to try:

    - **Rewire % → 100** turns the network into a random graph with the
      same number of edges. Clustering and assortativity collapse toward
      zero, and the path length usually shrinks (random shortcuts).
    - **Edges %** away from 100 adds or removes edges. Above 100 raises
      density and shrinks diameter; well below 100 eventually breaks the
      giant component apart.
    - **Clustering bias** closes triangles (positive) or breaks them
      apart (negative). Watch global clustering rise or fall.
    - **Assortativity bias** does degree-preserving swaps that push
      similar-degree nodes together (positive) or apart (negative). The
      degree distribution stays the same.
    """)
    return


@app.cell
def s4_widgets(mo):
    rewire_pct = mo.ui.slider(
        start=0, stop=100, step=5, value=0,
        label="Rewire %",
        show_value=True,
        full_width=True,
    )
    edges_pct = mo.ui.slider(
        start=0, stop=200, step=5, value=100,
        label="Edges %  (100 = unchanged)",
        show_value=True,
        full_width=True,
    )
    clustering_bias = mo.ui.slider(
        start=-100, stop=100, step=10, value=0,
        label="Clustering bias  (+ closes triangles, − breaks them)",
        show_value=True,
        full_width=True,
    )
    assort_bias = mo.ui.slider(
        start=-100, stop=100, step=10, value=0,
        label="Assortativity bias  (degree-preserving swaps)",
        show_value=True,
        full_width=True,
    )
    mo.hstack(
        [rewire_pct, edges_pct, clustering_bias, assort_bias],
        gap=1.5, widths="equal",
    )
    return assort_bias, clustering_bias, edges_pct, rewire_pct


@app.cell
def s4_modify(assort_bias, clustering_bias, edges_pct, g, rewire_pct, rnd_mod):
    # All operations are seeded so the modified graph is deterministic.
    _rng = rnd_mod.Random(20260519)
    g_mod = g.copy()
    _n = g_mod.vcount()
    _orig_m = g.ecount()

    def _key(u, v):
        return (u, v) if u < v else (v, u)

    # 1) Rewire fraction of edges (keep one endpoint, randomize the other)
    _p = rewire_pct.value / 100.0
    if _p > 0 and g_mod.ecount() > 0:
        _ids = list(range(g_mod.ecount()))
        _rng.shuffle(_ids)
        _n_rewire = int(_p * len(_ids))
        _to_replace = sorted(_ids[:_n_rewire])
        if _to_replace:
            _existing = set(_key(e.source, e.target) for e in g_mod.es)
            _old_pairs = [
                (g_mod.es[i].source, g_mod.es[i].target) for i in _to_replace
            ]
            g_mod.delete_edges(_to_replace)
            _new_edges = []
            for _u, _v in _old_pairs:
                _keep = _u if _rng.random() < 0.5 else _v
                for _ in range(40):
                    _w = _rng.randrange(_n)
                    if _w != _keep and _key(_keep, _w) not in _existing:
                        _existing.add(_key(_keep, _w))
                        _new_edges.append((_keep, _w))
                        break
            if _new_edges:
                g_mod.add_edges(_new_edges)

    # 2) Target edge count
    _target_m = int(_orig_m * edges_pct.value / 100)
    _target_m = max(0, min(_n * (_n - 1) // 2, _target_m))
    if _target_m < g_mod.ecount():
        _ids = list(range(g_mod.ecount()))
        _rng.shuffle(_ids)
        g_mod.delete_edges(_ids[:g_mod.ecount() - _target_m])
    elif _target_m > g_mod.ecount():
        _n_add = _target_m - g_mod.ecount()
        _existing = set(_key(e.source, e.target) for e in g_mod.es)
        _new_edges = []
        _attempts = 0
        while len(_new_edges) < _n_add and _attempts < _n_add * 40 + 500:
            _u = _rng.randrange(_n)
            _v = _rng.randrange(_n)
            if _u != _v and _key(_u, _v) not in _existing:
                _existing.add(_key(_u, _v))
                _new_edges.append((_u, _v))
            _attempts += 1
        if _new_edges:
            g_mod.add_edges(_new_edges)

    # Helper: degree-preserving swap (a,b)(c,d) -> (a,c)(b,d) or (a,d)(b,c).
    # Score function decides which swap (if any) to take.
    def _swap_pass(n_ops, score_fn):
        _nbrs = [set(g_mod.neighbors(_i)) for _i in range(g_mod.vcount())]
        for _ in range(n_ops):
            _m_now = g_mod.ecount()
            if _m_now < 2:
                break
            _i1 = _rng.randrange(_m_now)
            _i2 = _rng.randrange(_m_now)
            if _i1 == _i2:
                continue
            _a = g_mod.es[_i1].source
            _b = g_mod.es[_i1].target
            _c = g_mod.es[_i2].source
            _d = g_mod.es[_i2].target
            if len({_a, _b, _c, _d}) < 4:
                continue
            _curr, _opt1, _opt2 = score_fn(_a, _b, _c, _d, _nbrs)
            # Highest score wins; ties stay
            _pick = max(
                (_curr, None), (_opt1, "ac"), (_opt2, "ad"),
                key=lambda t: t[0],
            )
            if _pick[1] is None or _pick[0] <= _curr:
                continue
            if _pick[1] == "ac":
                if g_mod.are_adjacent(_a, _c) or g_mod.are_adjacent(_b, _d):
                    continue
                g_mod.delete_edges([_i1, _i2])
                g_mod.add_edges([(_a, _c), (_b, _d)])
                _nbrs[_a].discard(_b); _nbrs[_b].discard(_a)
                _nbrs[_c].discard(_d); _nbrs[_d].discard(_c)
                _nbrs[_a].add(_c); _nbrs[_c].add(_a)
                _nbrs[_b].add(_d); _nbrs[_d].add(_b)
            else:
                if g_mod.are_adjacent(_a, _d) or g_mod.are_adjacent(_b, _c):
                    continue
                g_mod.delete_edges([_i1, _i2])
                g_mod.add_edges([(_a, _d), (_b, _c)])
                _nbrs[_a].discard(_b); _nbrs[_b].discard(_a)
                _nbrs[_c].discard(_d); _nbrs[_d].discard(_c)
                _nbrs[_a].add(_d); _nbrs[_d].add(_a)
                _nbrs[_b].add(_c); _nbrs[_c].add(_b)

    # 3) Clustering bias — swap that maximises/minimises triangle count.
    _cb = clustering_bias.value
    if _cb != 0 and g_mod.ecount() >= 2:
        _sign = 1 if _cb > 0 else -1
        _n_ops = int(abs(_cb) * 8)  # 100 -> 800 attempted swaps

        def _tri_score(a, b, c, d, nbrs_):
            # Score: number of triangles gained. Higher = more clustering.
            lost = len(nbrs_[a] & nbrs_[b]) + len(nbrs_[c] & nbrs_[d])
            gain_ac = len(nbrs_[a] & nbrs_[c]) + len(nbrs_[b] & nbrs_[d])
            gain_ad = len(nbrs_[a] & nbrs_[d]) + len(nbrs_[b] & nbrs_[c])
            curr = 0
            opt1 = _sign * (gain_ac - lost)
            opt2 = _sign * (gain_ad - lost)
            return curr, opt1, opt2

        _swap_pass(_n_ops, _tri_score)

    # 4) Assortativity bias — degree-preserving swaps.
    _ab = assort_bias.value
    if _ab != 0 and g_mod.ecount() >= 2:
        _sign = 1 if _ab > 0 else -1
        _n_ops = int(abs(_ab) * 8)
        _deg = g_mod.degree()

        def _assort_score(a, b, c, d, nbrs_):
            # Positive sign wants small degree-diff (similar paired).
            # Express as score that is HIGHER when diff is SMALLER.
            curr = -(abs(_deg[a] - _deg[b]) + abs(_deg[c] - _deg[d]))
            o1 = -(abs(_deg[a] - _deg[c]) + abs(_deg[b] - _deg[d]))
            o2 = -(abs(_deg[a] - _deg[d]) + abs(_deg[b] - _deg[c]))
            return _sign * curr, _sign * o1, _sign * o2

        _swap_pass(_n_ops, _assort_score)

    g_mod.simplify(multiple=True, loops=True)
    return (g_mod,)


@app.cell
def s4_metrics_and_viz(EDGE_COLOR, NEUTRAL_NODE, g, g_mod, ig, layout_coords, mo, np, plt):
    def _metrics(graph):
        _n = graph.vcount()
        _m = graph.ecount()
        _density = graph.density()
        _avg_deg = 2 * _m / _n if _n else 0.0
        _trans = graph.transitivity_undirected(mode="zero")
        try:
            _ass = graph.assortativity_degree(directed=False)
            if _ass is None:
                _ass = float("nan")
        except Exception:
            _ass = float("nan")
        if graph.is_connected() and _n > 0:
            _diam = graph.diameter()
            _avg_path = graph.average_path_length()
            _ncomp = 1
        else:
            _comps = graph.connected_components()
            _ncomp = len(_comps)
            if _n > 0:
                _gc = _comps.giant()
                _diam = _gc.diameter()
                _avg_path = _gc.average_path_length()
            else:
                _diam = float("nan")
                _avg_path = float("nan")
        return _n, _m, _avg_deg, _density, _ncomp, _diam, _avg_path, _trans, _ass

    _o = _metrics(g)
    _md_vals = _metrics(g_mod)

    def _fmt(x):
        if isinstance(x, float):
            if np.isnan(x):
                return "—"
            return f"{x:.3f}"
        return str(x)

    def _delta(a, b):
        try:
            if isinstance(a, float) and np.isnan(a):
                return ""
            if isinstance(b, float) and np.isnan(b):
                return ""
            if abs(b - a) < 1e-9:
                return ""
            return " ↑" if b > a else " ↓"
        except Exception:
            return ""

    _labels = [
        "Nodes", "Edges", "Avg. degree", "Density", "Components",
        "Diameter", "Avg. path", "Global clustering", "Assortativity",
    ]
    _lines = ["| Metric | Original | Modified |", "|---|---|---|"]
    for _name, _a, _b in zip(_labels, _o, _md_vals):
        _lines.append(f"| {_name} | {_fmt(_a)} | {_fmt(_b)}{_delta(_a, _b)} |")
    _table_md = mo.md("\n".join(_lines))

    _fig, _ax = plt.subplots(figsize=(7.0, 5.5))
    _ax.set_facecolor("white")
    _ax.grid(False)
    _ax.set_xticks([])
    _ax.set_yticks([])
    for _side in ("top", "right", "bottom", "left"):
        _ax.spines[_side].set_visible(False)
    ig.plot(
        g_mod,
        target=_ax,
        layout=layout_coords,
        vertex_color=NEUTRAL_NODE,
        vertex_size=18,
        vertex_frame_width=0,
        vertex_label=[""] * g_mod.vcount(),
        edge_color=EDGE_COLOR,
        edge_width=1.0,
    )
    _ax.set_title(f"Modified network  ·  n={g_mod.vcount()}, m={g_mod.ecount()}")
    plt.tight_layout()

    mo.hstack(
        [_table_md, _fig],
        widths=[1.0, 1.4],
        gap=1.5,
        align="start",
    )
    return


# -----------------------------------------------------------------------------
# Section 5 — Centrality (four panels, side by side)
# -----------------------------------------------------------------------------


@app.cell
def section5_header(mo):
    mo.md(r"""
    ---
    ## 5 · Rank it — four definitions of *important*

    A "central" node is the one that matters most — but what *matters*
    means depends on the question you're asking. Four classic answers:

    - **Degree** — most neighbours. Local influence.
    - **Betweenness** — sits on the most shortest paths. A bridge.
    - **Closeness** — short average distance to everyone else. A
      well-placed observer.
    - **Eigenvector** — connected to other well-connected nodes. Status by
      association.

    Same graph, four pictures. The top-ranked node in each panel gets a
    red ring; the table underneath lists the top three. Compare the four
    — they usually disagree, and that disagreement *is* the lesson.
    """)
    return


@app.cell
def s5_compute(g, np):
    def _safe(fn, fallback):
        try:
            v = np.array(fn(), dtype=float)
            if not np.all(np.isfinite(v)):
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return v
        except Exception:
            return np.array(fallback, dtype=float)

    def _eigenvector_safe():
        if g.is_connected() or g.vcount() == 0:
            return g.eigenvector_centrality()
        _comps = g.connected_components()
        _gc_ids = max(_comps, key=len)
        _gc = g.subgraph(_gc_ids)
        _vals = _gc.eigenvector_centrality()
        _out = [0.0] * g.vcount()
        for _i, _node in enumerate(_gc_ids):
            _out[_node] = _vals[_i]
        return _out

    centralities = {
        "Degree": np.array(g.degree(), dtype=float),
        "Betweenness": _safe(lambda: g.betweenness(), [0] * g.vcount()),
        "Closeness": _safe(lambda: g.closeness(), [0] * g.vcount()),
        "Eigenvector": _safe(_eigenvector_safe, [0] * g.vcount()),
    }
    fr_coords = [
        tuple(row) for row in g.layout_fruchterman_reingold(niter=500).coords
    ]
    return centralities, fr_coords


@app.cell
def s5_grid(EDGE_COLOR, centralities, fr_coords, g, ig, mo, np, plt):
    import matplotlib as _mpl

    _measures = ["Degree", "Betweenness", "Closeness", "Eigenvector"]
    # afmhot_r: light → dark as t increases. We sample 10 evenly-spaced
    # tones avoiding pure white at one end and pure black at the other,
    # so the highest decile is dark and the lowest is pale.
    _cmap_src = plt.get_cmap("afmhot_r")
    _decile_colors = [_cmap_src(0.05 + i * 0.10) for i in range(10)]
    _listed = _mpl.colors.ListedColormap(_decile_colors)
    _norm = _mpl.colors.BoundaryNorm(np.arange(11) - 0.5, 10)

    _names = g.vs["name"]
    _show_labels = g.vcount() <= 30

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 9.0), constrained_layout=True)

    # Discrete colour-bar legend ABOVE the grid.
    _sm = _mpl.cm.ScalarMappable(cmap=_listed, norm=_norm)
    _sm.set_array([])
    _cbar = _fig.colorbar(
        _sm, ax=_axes.ravel().tolist(),
        orientation="horizontal",
        location="top",
        shrink=0.6,
        aspect=40,
        ticks=np.arange(10),
    )
    _cbar.set_ticklabels([str(i + 1) for i in range(10)], fontsize=9)
    _cbar.set_label(
        "Centrality decile — D1 lowest (light) → D10 highest (dark), "
        "computed per panel",
        fontsize=10,
    )

    for _ax, _name in zip(_axes.flat, _measures):
        _c = np.asarray(centralities[_name], dtype=float)
        if np.any(_c != _c[0]):
            _bins = np.percentile(_c, np.arange(10, 100, 10))  # 9 cut-points
            _bin_idx = np.digitize(_c, _bins)  # 0..9
        else:
            _bin_idx = np.zeros(len(_c), dtype=int)
        _colors = [_decile_colors[int(b)] for b in _bin_idx]

        _ax.set_facecolor("white")
        _ax.grid(False)
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _side in ("top", "right", "bottom", "left"):
            _ax.spines[_side].set_visible(False)
        ig.plot(
            g,
            target=_ax,
            layout=fr_coords,
            vertex_color=_colors,
            vertex_size=14,
            vertex_frame_width=0,
            vertex_label=_names if _show_labels else [""] * g.vcount(),
            vertex_label_size=8,
            edge_color=EDGE_COLOR,
            edge_width=0.9,
        )
        _top = int(np.argmax(_c))
        _x, _y = fr_coords[_top]
        _ax.scatter(
            [_x], [_y], s=380, facecolors="none",
            edgecolors="#c0223b", linewidths=2.2, zorder=10,
        )
        _ax.set_title(
            f"{_name}  ·  top: {_names[_top]} ({_c[_top]:.3g})",
            fontsize=11,
        )

    mo.vstack([_fig])
    return


@app.cell
def s5_top3(centralities, g, mo, np):
    _names = g.vs["name"]
    _lines = ["| Centrality | 1st | 2nd | 3rd |", "|---|---|---|---|"]
    for _name in ["Degree", "Betweenness", "Closeness", "Eigenvector"]:
        _c = centralities[_name]
        _order = np.argsort(-_c)[:3]
        _top = " | ".join(
            f"{_names[i]} ({_c[i]:.3g})" for i in _order
        )
        _lines.append(f"| {_name} | {_top} |")
    mo.md("\n".join(_lines))
    return


@app.cell
def footer(mo):
    mo.md(r"""
    ---
    Network Science Summer School 2026 · Utrecht University ·
    standalone marimo companion app.
    """)
    return


if __name__ == "__main__":
    app.run()
