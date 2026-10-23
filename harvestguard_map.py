"""District geometry and relative vulnerability map; no external map service."""

import plotly.express as px
import plotly.graph_objects as go


def build_map(view, geo, selected_district):
    quantity = "predicted_loss_kg" in view
    score_column = "predicted_loss_kg" if quantity else "predicted_loss_pct"
    map_view = view[[
        "district",
        "predicted_loss_pct",
        "observed_loss_pct",
        "n_profiles",
        "display_rank",
        "support_level",
    ]].copy()
    if quantity:
        map_view["predicted_loss_kg"] = view["predicted_loss_kg"]
    risk_lookup = map_view.set_index("district").to_dict("index")

    priority_n = min(5, len(map_view))
    priority_names = set(
        map_view.loc[map_view.display_rank <= priority_n, "district"].tolist()
    )
    priority_view = map_view[map_view["district"].isin(priority_names)].copy()


    def _walk_points(obj):
        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2 and isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float)):
                yield float(obj[0]), float(obj[1])
            else:
                for item in obj:
                    yield from _walk_points(item)


    def _ring_area_and_centroid(ring):
        pts = [(float(pt[0]), float(pt[1])) for pt in ring if len(pt) >= 2]
        if len(pts) < 3:
            xs = [p[0] for p in pts] or [0.0]
            ys = [p[1] for p in pts] or [0.0]
            return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        a2 = cx6 = cy6 = 0.0
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            cross = x0 * y1 - x1 * y0
            a2 += cross
            cx6 += (x0 + x1) * cross
            cy6 += (y0 + y1) * cross
        if abs(a2) < 1e-12:
            xs = [p[0] for p in pts[:-1]]
            ys = [p[1] for p in pts[:-1]]
            return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
        return abs(a2) / 2.0, cx6 / (3.0 * a2), cy6 / (3.0 * a2)


    def _feature_label_point(feature):
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type")
        rings = []
        if gtype == "Polygon" and coords:
            rings = [coords[0]]
        elif gtype == "MultiPolygon":
            rings = [poly[0] for poly in coords if poly]
        if not rings:
            pts = list(_walk_points(coords))
            if not pts:
                return None
            return sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
        candidates = [_ring_area_and_centroid(r) for r in rings]
        _, cx, cy = max(candidates, key=lambda t: t[0])
        return cx, cy


    LABEL_SHIFT = {
        "Nyarugenge": (0.015, 0.030),
        "Kicukiro": (0.060, -0.042),
        "Gasabo": (0.070, 0.048),
        "Kamonyi": (-0.075, -0.060),
        "Rulindo": (0.025, 0.030),
        "Gakenke": (-0.020, 0.024),
        "Ngororero": (-0.038, 0.005),
        "Muhanga": (-0.018, -0.018),
    }

    all_points = list(
        _walk_points([f.get("geometry", {}).get("coordinates", []) for f in geo["features"]])
    )
    lon_min = min(x for x, _ in all_points)
    lon_max = max(x for x, _ in all_points)
    lat_min = min(y for _, y in all_points)
    lat_max = max(y for _, y in all_points)
    lon_pad = (lon_max - lon_min) * 0.035
    lat_pad = (lat_max - lat_min) * 0.035

    fig = go.Figure()

    if not priority_view.empty:
        risk_min = float(priority_view[score_column].min())
        risk_max = float(priority_view[score_column].max())
    else:
        risk_min = risk_max = 0.0


    def _priority_fill(name):
        if name not in priority_names:
            return "#e2e8d1" if name in risk_lookup else "#e4e2db"
        val = float(risk_lookup[name][score_column])
        if risk_max <= risk_min + 1e-12:
            t = 0.72
        else:
            t = (val - risk_min) / (risk_max - risk_min)
        t = 0.20 + 0.80 * max(0.0, min(1.0, t))
        return px.colors.sample_colorscale(
            [[0.0, "#f4e5ba"], [0.45, "#edc98c"], [0.75, "#e4ac6d"], [1.0, "#d58d56"]],
            [t],
        )[0]


    for feature in geo["features"]:
        name = feature["properties"]["shapeName"]
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        gtype = geom.get("type")

        if gtype == "Polygon":
            polygon_parts = [coords]
        elif gtype == "MultiPolygon":
            polygon_parts = coords
        else:
            continue

        row = risk_lookup.get(name)
        is_priority = name in priority_names
        if row is None:
            hover_text = f"<b>{name}</b><br>No prediction for the selected crop/season."
        else:
            tier = "TOP-5 BY ESTIMATED KG LOST" if quantity and is_priority else "TOP-5 BY LOSS RATE" if is_priority else "Outside current Top-5"
            amount = f"Estimated loss: <b>{float(row['predicted_loss_kg']):,.2f} kg</b><br>" if quantity else ""
            hover_text = (
                f"<b>{name}</b><br>"
                f"{tier}<br>"
                f"{amount}"
                f"Predicted loss rate: <b>{float(row['predicted_loss_pct']):.2f}%</b><br>"
                f"Rank: #{int(row['display_rank'])}<br>"
                f"Profiles: {int(row['n_profiles']):,}<br>"
                f"Sample support: {row['support_level']}<br>"
                f"Observed 2025 loss: {float(row['observed_loss_pct']):.2f}%"
            )

        for polygon in polygon_parts:
            if not polygon:
                continue
            outer_ring = polygon[0]
            if not outer_ring:
                continue
            xs = [float(pt[0]) for pt in outer_ring]
            ys = [float(pt[1]) for pt in outer_ring]
            if xs[0] != xs[-1] or ys[0] != ys[-1]:
                xs.append(xs[0])
                ys.append(ys[0])

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    fill="toself",
                    fillcolor=_priority_fill(name),
                    line=dict(color="#214b32" if name == selected_district else "#758064", width=3.5 if name == selected_district else 1.0),
                    text=[hover_text] * len(xs),
                    hovertemplate="%{text}<extra></extra>",
                    hoveron="fills+points",
                    showlegend=False,
                    name=name,
                )
            )

    normal_label_x, normal_label_y, normal_label_text = [], [], []
    priority_label_x, priority_label_y, priority_label_text = [], [], []

    for feature in geo["features"]:
        name = feature["properties"]["shapeName"]
        point = _feature_label_point(feature)
        if point is None:
            continue
        x, y = point
        dx, dy = LABEL_SHIFT.get(name, (0.0, 0.0))
        x += dx
        y += dy

        if name in priority_names:
            priority_label_x.append(x)
            priority_label_y.append(y)
            priority_label_text.append(f"<b>{name}</b>")
        else:
            normal_label_x.append(x)
            normal_label_y.append(y)
            normal_label_text.append(name)

    fig.add_trace(
        go.Scatter(
            x=normal_label_x,
            y=normal_label_y,
            text=normal_label_text,
            mode="text",
            textfont=dict(size=9, color="#3a4a35", family="Arial"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=priority_label_x,
            y=priority_label_y,
            text=priority_label_text,
            mode="text",
            textfont=dict(size=11, color="#111111", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_xaxes(
        range=[lon_min - lon_pad, lon_max + lon_pad],
        visible=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[lat_min - lat_pad, lat_max + lat_pad],
        visible=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        height=555,
        margin=dict(l=6, r=6, t=8, b=6),
        paper_bgcolor="#f6f5ed",
        plot_bgcolor="#f6f5ed",
        font=dict(color="#203b2e"),
        hoverlabel=dict(
            bgcolor="#f6f5ed",
            font_color="#203b2e",
            bordercolor="#111111",
            font_size=13,
        ),
        dragmode=False,
    )
    return fig
