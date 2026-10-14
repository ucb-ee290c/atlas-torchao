"""Offline, descriptive native-action CSV/SVG reports from verified recordings.

No model loading, fitting, physical-unit conversion, task success, performance or
acceptance threshold is inferred. Charts are computed from the verified raw pairs,
not a separately supplied summary. SVG rendering uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import characterize_actions as actions

SCHEMA = "atlas.action_characterization.report.v1"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#222222")


def xml_text(value):
    """Escape labels, replacing characters not permitted in XML 1.0."""
    text = str(value)
    text = "".join(c if c in "\t\n\r" or 0x20 <= ord(c) <= 0xD7FF
                   or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF
                   else "\ufffd" for c in text)
    return html.escape(text, quote=True)


def csv_text(value):
    # Spreadsheet consumers must not execute untrusted channel/task labels.
    if isinstance(value, str) and (value.lstrip().startswith(("=", "+", "-", "@"))
                                  or value.startswith(("\t", "\r", "\n"))):
        return "'" + value
    return value


def write_csv(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_text(value) for key, value in row.items()})


def metric_rows(summary):
    """Flatten every native comparison and task/episode grouping without pooling units."""
    groups = [("all", {}, {name: summary[name] for name in summary["per_chunk_components"]["measurements"]})]
    for axis in ("tasks", "episodes"):
        groups += [(axis, group, group["measurements"]) for group in summary["groups"][axis]]
    channels = summary["contract"]["channels"]
    units = summary["contract"].get("units", ["unknown"] * len(channels))
    for axis, group, measurements in groups:
        for name, measurement in measurements.items():
            entries = [("pooled", None, None, measurement["pooled"])]
            entries += [("component", None, k, value) for k, value in enumerate(measurement["per_component"])]
            entries += [("horizon_component", t, k, value)
                        for t, step in enumerate(measurement["per_horizon_component"]) for k, value in enumerate(step)]
            for point, horizon, component, value in entries:
                yield {"group": axis, "task_id": group.get("task_id"), "episode_id": group.get("episode_id"),
                       "comparison": name, "point": point, "horizon_index": horizon, "component_index": component,
                       "channel": channels[component] if component is not None else "pooled_native_components",
                       "units": ("model_space" if name.startswith("model_") else
                                 units[component] if component is not None else "mixed_native_components"),
                       **{key: value[key] for key in ("count", "bias", "rmse", "mae", "std", "variance", "max_abs")},
                       **{"abs_q" + key: (value["absolute_quantiles"] or {}).get(key)
                          for key in ("0.5", "0.9", "0.95", "0.99")}}


def chunk_rows(summary):
    for name, measurement in summary["per_chunk_components"]["measurements"].items():
        for identity, rmses, steps in zip(summary["per_chunk_components"]["observation_ids"],
                                         measurement["rmse"], measurement["valid_steps"]):
            for k, value in enumerate(rmses):
                yield {**identity, "comparison": name, "component_index": k,
                       "channel": summary["contract"]["channels"][k], "steps": steps, "rmse": value,
                       "units": "model_space" if name.startswith("model_") else
                           summary["contract"].get("units", ["unknown"] * summary["contract"]["dimension"])[k]}


def chunk_distribution_rows(summary):
    groups = [("all", {}, summary["per_chunk_components"]["measurements"])]
    for axis in ("tasks", "episodes"):
        groups += [(axis, group, {name: value["per_chunk_components"]
                    for name, value in group["measurements"].items()}) for group in summary["groups"][axis]]
    for axis, group, measurements in groups:
        for name, value in measurements.items():
            for k, distribution in enumerate(value["per_component_rmse_distribution"]):
                yield {"group": axis, "task_id": group.get("task_id"), "episode_id": group.get("episode_id"),
                       "comparison": name, "component_index": k, "channel": summary["contract"]["channels"][k],
                       "units": "model_space" if name.startswith("model_") else
                           summary["contract"].get("units", ["unknown"] * summary["contract"]["dimension"])[k],
                       **{key: distribution[key] for key in ("count", "mean", "std", "variance", "min", "max")},
                       **{"q" + key: (distribution["quantiles"] or {}).get(key)
                          for key in ("0.5", "0.9", "0.95", "0.99")}}


def chart_rows(summary, measurements, *, ecdf_points):
    """Chart source values; ECDF ranks are exact, visual sampling is deterministic."""
    import numpy as np

    for name, (error, mask) in measurements.items():
        measurement = summary[name]
        for k, value in enumerate(measurement["per_component"]):
            yield {"comparison": name, "chart": "component_rmse", "component_index": k,
                   "horizon_index": None, "count": value["count"], "value": value["rmse"], "cdf": None}
        for t, step in enumerate(measurement["per_horizon_component"]):
            for k, value in enumerate(step):
                yield {"comparison": name, "chart": "horizon_component_rmse", "component_index": k,
                       "horizon_index": t, "count": value["count"], "value": value["rmse"], "cdf": None}
        for k in range(error.shape[2]):
            values = np.sort(np.abs(error[:, :, k][mask]))
            if not len(values):
                yield {"comparison": name, "chart": "absolute_error_ecdf", "component_index": k,
                       "horizon_index": None, "count": 0, "value": None, "cdf": None}
                continue
            ranks = np.unique(np.linspace(0, len(values) - 1, min(ecdf_points, len(values)), dtype=int))
            for rank in ranks:
                value = float(values[rank])
                yield {"comparison": name, "chart": "absolute_error_ecdf", "component_index": k,
                       "horizon_index": None, "count": int(len(values)), "value": value,
                       "cdf": float(np.searchsorted(values, value, side="right") / len(values))}


def svg_document(title, content, source_hash):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="620" viewBox="0 0 960 620">'
            '<rect width="960" height="620" fill="white"/>'
            f'<metadata>chart_data.csv sha256={source_hash}; descriptive native-action comparison</metadata>'
            '<g font-family="sans-serif" font-size="13">'
            f'<text x="40" y="30" font-size="18">{xml_text(title)}</text>'
            '<text x="40" y="55">Descriptive only; overlapping chunks are not independent; no acceptance cutoff.</text>'
            + content
            + f'<text x="40" y="610">Space: {"model-space" if title.startswith("model_") else "native dataset units"}.'
              ' No physical conversion is performed.</text></g></svg>\n')


def component_svg(rows, channels):
    values = [row["value"] for row in rows if row["value"] is not None]
    scale = max(values, default=0) or 1
    content = '<text x="40" y="82">Component RMSE (native model/dataset units; units may differ by component)</text>'
    width = 760 / len(channels)
    for row in rows:
        k, value = row["component_index"], row["value"]
        x = 110 + k * width
        height = 340 * (value or 0) / scale
        content += (f'<rect x="{x:.3f}" y="{450-height:.3f}" width="{width*.65:.3f}" '
                    f'height="{height:.3f}" fill="{COLORS[k % len(COLORS)]}"/>'
                    f'<text x="{x:.3f}" y="{435-height:.3f}">{value:.6g}</text>' if value is not None
                    else f'<text x="{x:.3f}" y="430">no valid targets</text>')
        content += f'<text transform="translate({x:.3f},470) rotate(30)">{xml_text(channels[k])}</text>'
    return content


def heatmap_svg(rows, channels, horizon):
    values = [row["value"] for row in rows if row["value"] is not None]
    maximum = max(values, default=0) or 1
    width, height = 760 / horizon, 360 / len(channels)
    content = '<text x="40" y="82">Horizon/component RMSE; missing target population is gray</text>'
    for row in rows:
        k, t, value = row["component_index"], row["horizon_index"], row["value"]
        color = "#dddddd" if value is None else f"rgb({255-int(180*value/maximum)},255,255)"
        content += (f'<rect x="{150+t*width:.3f}" y="{110+k*height:.3f}" width="{width:.3f}" '
                    f'height="{height:.3f}" fill="{color}"><title>{xml_text(channels[k])}; '
                    f'horizon={t}; n={row["count"]}; RMSE={value}</title></rect>')
    for k, channel in enumerate(channels):
        content += f'<text x="10" y="{120+k*height:.3f}">{xml_text(channel)}</text>'
    content += (f'<text x="150" y="495">Horizon index 0 → {horizon-1}; color scale 0 → {maximum:.6g}</text>'
                '<text x="150" y="520">Native component units retained; physical comparability is unknown.</text>')
    return content


def ecdf_svg(rows, channels, ecdf_points):
    maximum = max((row["value"] for row in rows if row["value"] is not None), default=0) or 1
    content = ('<text x="40" y="82">Absolute-error empirical CDF per native component</text>'
               '<path d="M110 100 V450 H890" stroke="#222" fill="none"/>'
               '<text x="40" y="105">1.0</text><text x="40" y="455">0.0</text>')
    for k, channel in enumerate(channels):
        points = [row for row in rows if row["component_index"] == k and row["value"] is not None]
        color = COLORS[k % len(COLORS)]
        if points:
            path = "M110 450" + "".join(f' H{110+780*row["value"]/maximum:.3f} V{450-350*row["cdf"]:.3f}'
                                         for row in points)
            content += f'<path d="{path}" stroke="{color}" fill="none"/>'
        content += (f'<text x="{110+(k % 3)*270}" y="{505+(k//3)*20}" fill="{color}">'
                    f'{xml_text(channel)}{(" (empty)" if not points else "")}</text>')
    content += (f'<text x="110" y="480">Absolute error 0 → {maximum:.6g} (native units)</text>'
                f'<text x="40" y="590">At most {ecdf_points} ranks/component; exact CDF at each shown value.</text>')
    return content


def report(record_dir, out_dir, *, ecdf_points=256):
    if not isinstance(ecdf_points, int) or not 2 <= ecdf_points <= 4096:
        raise ValueError("ecdf_points must be an integer from 2 through 4096")
    rows, header, completion, verification = actions.load_verified_recording(record_dir)
    contract = header["input_manifest"]["action_contract"]
    summary = actions.analyze_records(rows, contract)
    provenance = {"record_directory_hint": str(Path(record_dir).resolve()),
                  "completion_sha256": completion["verified_completion_sha256"],
                  "header_sha256": completion["header_sha256"], "actions_sha256": completion["actions_sha256"],
                  "input_manifest_sha256": header["input_manifest_sha256"],
                  "calibration_verification_sha256": completion["calibration_verification_sha256"],
                  "recording_source_commit": header.get("source_commit"),
                  "candidate_recipe": header.get("candidate_recipe"), "candidate_mode": header.get("candidate_mode")}
    summary.update(provenance=provenance, calibration_verification=verification,
                   layer_trace=completion.get("layer_trace"))
    arrays, mask = actions.validated_arrays(rows, contract)
    data = list(chart_rows(summary, actions.measurement_errors(arrays, mask), ecdf_points=ecdf_points))
    # Computed overflow must refuse before creating an output directory.
    json.dumps([summary, data], allow_nan=False)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=False)
    actions.write_new(output / "summary.json", summary)
    metric_fields = ["group", "task_id", "episode_id", "comparison", "point", "horizon_index", "component_index",
                     "channel", "units", "count", "bias", "rmse", "mae", "std", "variance", "max_abs",
                     "abs_q0.5", "abs_q0.9", "abs_q0.95", "abs_q0.99"]
    write_csv(output / "metrics.csv", metric_fields, metric_rows(summary))
    write_csv(output / "per_chunk_rmse.csv", ["task_id", "episode_id", "frame_index", "sampler_seed", "comparison",
              "component_index", "channel", "steps", "rmse", "units"], chunk_rows(summary))
    write_csv(output / "chunk_rmse_distributions.csv", ["group", "task_id", "episode_id", "comparison",
              "component_index", "channel", "units", "count", "mean", "std", "variance", "min", "max",
              "q0.5", "q0.9", "q0.95", "q0.99"], chunk_distribution_rows(summary))
    write_csv(output / "chart_data.csv", ["comparison", "chart", "component_index", "horizon_index", "count",
                                         "value", "cdf"], data)
    source_hash = actions.digest(output / "chart_data.csv")
    charts = []
    for comparison in summary["per_chunk_components"]["measurements"]:
        for chart in ("component_rmse", "horizon_component_rmse", "absolute_error_ecdf"):
            selected = [row for row in data if row["comparison"] == comparison and row["chart"] == chart]
            if chart == "component_rmse":
                content = component_svg(selected, contract["channels"])
            elif chart == "horizon_component_rmse":
                content = heatmap_svg(selected, contract["channels"], contract["returned_horizon"])
            else:
                content = ecdf_svg(selected, contract["channels"], ecdf_points)
            filename = comparison + "__" + chart + ".svg"
            with (output / filename).open("x", encoding="utf-8") as stream:
                stream.write(svg_document(comparison + " / " + chart, content, source_hash))
            charts.append({"file": filename, "comparison": comparison, "chart": chart,
                           "chart_source_file": "chart_data.csv", "chart_source_sha256": source_hash})
    manifest = {"schema": SCHEMA, "complete": True, "provenance": provenance,
                "files_sha256": {path.name: actions.digest(path) for path in sorted(output.iterdir())},
                "charts": charts, "ecdf_points_max_per_component": ecdf_points,
                "report_source_sha256": {Path(__file__).name: actions.digest(__file__),
                                         Path(actions.__file__).name: actions.digest(actions.__file__)},
                "scope": {"descriptive_only": True, "physical_unit_conversion": False,
                    "acceptance_thresholds": None, "task_success_measured": False, "performance_measured": False,
                    "dependence": "chunks can overlap; scalar steps/chunks are not claimed independent",
                    "units": "model and native dataset units remain separate; unknown stays unknown",
                    "task_grouping": "recorded task IDs, not closed-loop task success",
                    "label_rendering": "XML controls replaced; spreadsheet-leading formulas quoted in CSV",
                    "integrity": "hash binding, not an authenticity certificate",
                    "original_manifest_bytes": "original byte hash recorded in header, not independently replayed",
                    "embedded_manifest_content": "canonical content digest verified when supplied"}}
    actions.write_new(output / "report_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ecdf-points", type=int, default=256)
    args = parser.parse_args()
    try:
        result = report(args.record_dir, args.out_dir, ecdf_points=args.ecdf_points)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"schema": result["schema"], "complete": result["complete"], "charts": len(result["charts"])}))


if __name__ == "__main__":
    main()
