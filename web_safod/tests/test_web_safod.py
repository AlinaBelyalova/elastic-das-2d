from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from web_safod.builder.assets import ensure_web_plots
from web_safod.builder.catalog import load_catalog
from web_safod.builder.geometry import export_safod_geojson
from web_safod.builder.validation import validate_site


class WebSafodTests(unittest.TestCase):
    def test_catalog_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.csv"
            pd.DataFrame(
                [
                    {
                        "event_id": "nc1",
                        "origin_time_utc": "2026-01-01T00:00:00Z",
                        "latitude": 35.9,
                        "longitude": -120.5,
                        "depth_km": 3.2,
                        "magnitude": 1.1,
                    }
                ]
            ).to_csv(path, index=False)
            frame = load_catalog(path)
            self.assertEqual(frame.iloc[0]["event_id"], "nc1")

    def test_geometry_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            xlsx = tmp / "geometry.xlsx"
            out = tmp / "geometry.geojson"
            pd.DataFrame(
                [
                    {
                        "Channel": 1,
                        "Section": "Down-leg",
                        "MD_m": 0,
                        "TVD_m": 0,
                        "Lat_WGS84": 35.97,
                        "Lon_WGS84": -120.55,
                    },
                    {
                        "Channel": 2,
                        "Section": "Down-leg",
                        "MD_m": 100,
                        "TVD_m": 100,
                        "Lat_WGS84": 35.971,
                        "Lon_WGS84": -120.549,
                    },
                    {
                        "Channel": 3,
                        "Section": "Up-leg",
                        "MD_m": 100,
                        "TVD_m": 100,
                        "Lat_WGS84": 35.971,
                        "Lon_WGS84": -120.549,
                    },
                    {
                        "Channel": 4,
                        "Section": "Up-leg",
                        "MD_m": 0,
                        "TVD_m": 0,
                        "Lat_WGS84": 35.97,
                        "Lon_WGS84": -120.55,
                    },
                ]
            ).to_excel(xlsx, index=False)
            result = export_safod_geojson(xlsx, out)
            self.assertEqual(result["type"], "FeatureCollection")
            self.assertTrue(out.exists())
            self.assertGreaterEqual(len(result["features"]), 3)

    def test_webp_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            repo = tmp / "repo"
            event_qc = repo / "results/safod_deep/event_qc"
            meta_dir = event_qc / "metadata"
            meta_dir.mkdir(parents=True)
            source = event_qc / "plot.png"
            Image.new("RGB", (800, 400), "white").save(source)

            meta_path = meta_dir / "plot.json"
            meta = {
                "event_id": "nc1",
                "image_path": str(
                    Path("results/safod_deep/event_qc/plot.png")
                ),
                "recipe": {
                    "name": "near_5_30Hz",
                    "fmin_hz": 5,
                    "fmax_hz": 30,
                },
                "_metadata_path": str(meta_path),
            }
            output = repo / "web_safod/site/plots"
            cache = {}
            plots, warnings = ensure_web_plots(
                "nc1",
                [meta],
                repo_root=repo,
                event_qc_dir=event_qc,
                output_dir=output,
                cache=cache,
                max_long_edge_px=600,
                quality=80,
            )
            self.assertFalse(warnings)
            self.assertEqual(len(plots), 1)
            self.assertTrue((repo / "web_safod/site" / plots[0]["url"]).exists())

    def test_public_validation_rejects_private_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site / "data").mkdir()
            (site / "plots").mkdir()
            for name in ("index.html", "app.js", "style.css"):
                (site / name).write_text("x", encoding="utf-8")

            events = [
                {
                    "event_id": "nc1",
                    "latitude": 35.9,
                    "longitude": -120.5,
                    "depth_km": 3.0,
                    "plot_available": False,
                    "plots": [],
                    "debug": "/home/user/private/file.h5",
                }
            ]
            (site / "data/events.json").write_text(
                json.dumps(events),
                encoding="utf-8",
            )
            (site / "data/site_status.json").write_text(
                json.dumps({"recorded_events": 1, "plots_available": 0}),
                encoding="utf-8",
            )
            (site / "data/site_config.json").write_text(
                json.dumps({"build_id": "x"}),
                encoding="utf-8",
            )
            (site / "data/safod_geometry.geojson").write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {},
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [-120.5, 35.9],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            errors = validate_site(site)
            self.assertTrue(any("Private/raw path" in x for x in errors))


if __name__ == "__main__":
    unittest.main()
