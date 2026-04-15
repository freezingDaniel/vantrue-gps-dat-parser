import json
import os
import sys
from pathlib import Path

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISPLAY_STEP = 10            # Use one point every x seconds
MIN_MOVEMENT_KM = 0.1        # Suppress GPS noise when stationary
KNOTS_TO_KMH = 1.852
ALTITUDE_LABEL = "Altitude"
EARTH_RADIUS_KM = 6371.0
CSV_COLUMNS = ["timestamp", "lat", "lat_dir", "lon", "lon_dir", "speed_knots", "altitude"]
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
DISPLAY_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_PATH = os.environ.get("GPS_DATA_PATH", ".")

# Visibility tuning
POLYLINE_WEIGHT = 5          # px width of track segments
POLYLINE_OPACITY = 0.90
WAYPOINT_RADIUS = 5
WAYPOINT_OPACITY = 0.55      # outline opacity for non-boundary waypoints
BOUNDARY_MARKER_ICON = "map-marker"   # Font Awesome icon name (folium default prefix)


# ---------------------------------------------------------------------------
# Geo utilities
# ---------------------------------------------------------------------------
def haversine_series(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> np.ndarray:
    lat1r, lon1r = np.radians(lat1.values), np.radians(lon1.values)
    lat2r, lon2r = np.radians(lat2.values), np.radians(lon2.values)
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def compute_thresholded_distance(df: pd.DataFrame) -> tuple[pd.Series, float]:
    if len(df) == 1:
        return pd.Series([0.0], index=df.index), 0.0

    lat1 = df["lat"].iloc[:-1].reset_index(drop=True)
    lon1 = df["lon"].iloc[:-1].reset_index(drop=True)
    lat2 = df["lat"].iloc[1:].reset_index(drop=True)
    lon2 = df["lon"].iloc[1:].reset_index(drop=True)

    segment_km = haversine_series(lat1, lon1, lat2, lon2)
    counted = np.where(segment_km > MIN_MOVEMENT_KM, segment_km, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(counted)])
    return pd.Series(cumulative, index=df.index), float(counted.sum())


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_gps_csv(file_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    after_break = True

    with open(file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if rows:
                    rows[-1]["before_break"] = True
                after_break = True
                continue
            parts = stripped.split(",")
            if len(parts) < len(CSV_COLUMNS):
                continue
            record: dict = dict(zip(CSV_COLUMNS, parts[: len(CSV_COLUMNS)]))
            record["after_break"] = after_break
            record["before_break"] = False
            rows.append(record)
            after_break = False

    if not rows:
        raise ValueError("No valid GPS records found in the file.")

    rows[-1]["before_break"] = True

    df = pd.DataFrame(rows)

    for col in ["timestamp", "lat_dir", "lon_dir"]:
        df[col] = df[col].astype(str).str.strip()

    for col in ["lat", "lon", "speed_knots", "altitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time_str"] = pd.to_datetime(
        df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce"
    ).dt.strftime(DISPLAY_TIMESTAMP_FORMAT)

    df = df.dropna(subset=["lat", "lon", "time_str"]).copy()
    if df.empty:
        raise ValueError("No valid GPS records found after parsing.")

    df["lat"] = df["lat"] * np.where(df["lat_dir"] == "S", -1, 1)
    df["lon"] = df["lon"] * np.where(df["lon_dir"] == "W", -1, 1)
    df["speed_kmh"] = df["speed_knots"] * KNOTS_TO_KMH

    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def sample_track(df: pd.DataFrame, step: int = DISPLAY_STEP) -> pd.DataFrame:
    if len(df) <= 2:
        return df.copy()

    break_indices = set(df.index[df["after_break"] | df["before_break"]].tolist())
    step_indices = set(range(0, len(df), step))
    step_indices.add(len(df) - 1)
    all_indices = sorted(break_indices | step_indices)

    return df.iloc[all_indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Speed colormap
# ---------------------------------------------------------------------------
def _build_speed_colormap(df: pd.DataFrame) -> cm.LinearColormap:
    valid = df["speed_kmh"].dropna()
    vmin = float(valid.min()) if not valid.empty else 0.0
    vmax = float(valid.max()) if not valid.empty else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return cm.LinearColormap(
        colors=["blue", "cyan", "green", "yellow", "orange", "red"],
        vmin=vmin,
        vmax=vmax,
        caption="Speed (km/h)",
    )


# ---------------------------------------------------------------------------
# Folium map construction
# ---------------------------------------------------------------------------
def _speed_display(row) -> str:
    if pd.isna(row.speed_knots):
        return "N/A"
    if pd.isna(row.speed_kmh):
        return f"{row.speed_knots:.1f} knots"
    return f"{row.speed_kmh:.1f} km/h ({row.speed_knots:.1f} knots)"


def _popup_html(row, role: str) -> str:
    altitude = f"{row.altitude:.1f} m" if pd.notna(row.altitude) else "N/A"
    return (
        f"<b>{row.time_str}</b><br>"
        f"<i>{role}</i><br>"
        f"Lat: {row.lat:.6f}<br>"
        f"Lon: {row.lon:.6f}<br>"
        f"Speed: {_speed_display(row)}<br>"
        f"Dist: {row.cum_km:.3f} km<br>"
        f"{ALTITUDE_LABEL}: {altitude}"
    )


def build_base_map(df: pd.DataFrame) -> tuple[folium.Map, cm.LinearColormap]:
    """Create Folium map with speed-colored segments and break-aware markers."""
    m = folium.Map(
        location=[df["lat"].mean(), df["lon"].mean()],
        zoom_start=15,
        # CartoDB Positron gives a light, low-contrast basemap so the
        # coloured track stands out more than on OSM.
        tiles="CartoDB Positron",
    )
    # Add OSM as an optional overlay so users can switch
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    colormap = _build_speed_colormap(df)

    # ------------------------------------------------------------------
    # 1. Draw a thin dark shadow pass first for contrast on light tiles
    # ------------------------------------------------------------------
    seg_lats: list[float] = []
    seg_lons: list[float] = []
    seg_colors: list[float] = []

    def _flush_segment(shadow: bool = False) -> None:
        if len(seg_lats) >= 2:
            coords = list(zip(seg_lats, seg_lons))
            if shadow:
                folium.PolyLine(
                    locations=coords,
                    color="#333333",
                    weight=POLYLINE_WEIGHT + 3,
                    opacity=0.35,
                ).add_to(m)
            else:
                folium.ColorLine(
                    positions=coords,
                    colors=seg_colors,
                    colormap=colormap,
                    weight=POLYLINE_WEIGHT,
                    opacity=POLYLINE_OPACITY,
                ).add_to(m)
        seg_lats.clear()
        seg_lons.clear()
        seg_colors.clear()

    # Shadow pass
    for i, row in df.iterrows():
        if row["after_break"] and seg_lats:
            _flush_segment(shadow=True)
        seg_lats.append(float(row["lat"]))
        seg_lons.append(float(row["lon"]))
        seg_colors.append(float(row["speed_kmh"]) if pd.notna(row["speed_kmh"]) else 0.0)
        if row["before_break"]:
            _flush_segment(shadow=True)
    _flush_segment(shadow=True)

    # Colour pass
    for i, row in df.iterrows():
        if row["after_break"] and seg_lats:
            _flush_segment()
        seg_lats.append(float(row["lat"]))
        seg_lons.append(float(row["lon"]))
        seg_colors.append(float(row["speed_kmh"]) if pd.notna(row["speed_kmh"]) else 0.0)
        if row["before_break"]:
            _flush_segment()
    _flush_segment()

    colormap.add_to(m)

    # ------------------------------------------------------------------
    # 2. Markers
    #    - Segment start/end  → folium.Marker with colored icon (pin shape)
    #    - Waypoints          → small CircleMarker (faint, non-intrusive)
    # ------------------------------------------------------------------
    for row in df.itertuples(index=False):
        is_start = row.after_break
        is_end   = row.before_break
        is_boundary = is_start or is_end

        altitude   = f"{row.altitude:.1f} m" if pd.notna(row.altitude) else "N/A"
        role       = "Segment Start" if is_start else ("Segment End" if is_end else "Waypoint")
        speed_str  = _speed_display(row)
        popup_html = (
            f"<b>{row.time_str}</b><br>"
            f"<i>{role}</i><br>"
            f"Lat: {row.lat:.6f}<br>"
            f"Lon: {row.lon:.6f}<br>"
            f"Speed: {speed_str}<br>"
            f"Dist: {row.cum_km:.3f} km<br>"
            f"{ALTITUDE_LABEL}: {altitude}"
        )

        if is_boundary:
            # Colored pin marker — much easier to spot than a circle
            icon_color = "green" if is_start else "red"
            folium.Marker(
                location=[row.lat, row.lon],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{role} — {row.time_str}",
                icon=folium.Icon(
                    color=icon_color,
                    icon="play" if is_start else "stop",   # Font Awesome glyphs
                    prefix="fa",
                ),
            ).add_to(m)
        else:
            # Waypoint: subtle circle, visible but not distracting
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=WAYPOINT_RADIUS,
                color="#1a6fcc",
                fill=False,
                opacity=WAYPOINT_OPACITY,
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(m)

    return m, colormap


# ---------------------------------------------------------------------------
# HTML overlay generation
# ---------------------------------------------------------------------------
def _legend_html(total_km: float) -> str:
    return f"""
    <div style="position:fixed;bottom:85px;left:30px;z-index:1000;background:white;
                padding:10px;border:1px solid grey;border-radius:5px;font-size:13px;">
      <b>Legend</b><br>
      <span style="color:green;font-size:16px;">&#9679;</span> Segment Start<br>
      <span style="color:red;font-size:16px;">&#9679;</span> Segment End<br>
      <span style="color:#1a6fcc;font-size:16px;">&#9675;</span> Waypoint<br>
      <hr style="margin:6px 0;">
      <b>Total:</b> {total_km:.3f} km
    </div>
    """


def _slider_html(n_points: int) -> str:
    return f"""
    <div style="position:fixed;bottom:25px;left:50%;transform:translateX(-50%);
                z-index:1000;background:white;padding:10px 12px;border:1px solid grey;
                border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);
                width:80vw;font-family:sans-serif;">
      <div style="display:flex;align-items:center;gap:10px;position:relative;">
        <label for="timeSlider" style="font-weight:600;white-space:nowrap;">Time</label>
        <div style="position:relative;flex:1;min-width:0;">
          <input id="timeSlider" type="range" min="0" max="{n_points - 1}"
                 value="{n_points - 1}" step="1" style="width:100%;margin:0;">
          <div id="segmentMarkers"
               style="position:absolute;left:0;right:0;top:50%;height:10px;pointer-events:none;"></div>
        </div>
        <span id="timeLabel" style="white-space:nowrap;font-size:13px;"></span>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:12px;
                  color:#555;margin-top:4px;flex-wrap:wrap;gap:6px;">
        <span id="distLabel"></span>
        <span id="speedLabel"></span>
        <span id="altLabel"></span>
        <span id="roleLabel"></span>
      </div>
    </div>
    """


def _script_html(track_points: list[dict], map_name: str) -> str:
    return f"""
    <script>
    window.addEventListener('load', function() {{
        const map            = {map_name};
        const trackData      = {json.dumps(track_points)};
        const ALTITUDE_LABEL = {json.dumps(ALTITUDE_LABEL)};
        const slider         = document.getElementById('timeSlider');
        const timeLabel      = document.getElementById('timeLabel');
        const distLabel      = document.getElementById('distLabel');
        const speedLabel     = document.getElementById('speedLabel');
        const altLabel       = document.getElementById('altLabel');
        const roleLabel      = document.getElementById('roleLabel');
        const markerRail     = document.getElementById('segmentMarkers');

        const WAYPOINT_RADIUS  = {WAYPOINT_RADIUS};
        const WAYPOINT_OPACITY = {WAYPOINT_OPACITY};
        const POLYLINE_WEIGHT  = {POLYLINE_WEIGHT};
        const POLYLINE_OPACITY = {POLYLINE_OPACITY};

        const segmentStarts = trackData
            .map((p, idx) => p.after_break ? idx : null)
            .filter(idx => idx !== null);

        function renderSegmentMarkers() {{
            markerRail.innerHTML = '';
            if (trackData.length <= 1) return;

            segmentStarts.forEach((idx, i) => {{
                const dot = document.createElement('button');
                const leftPct = (idx / (trackData.length - 1)) * 100;

                dot.type = 'button';
                dot.title = 'Skip to segment start ' + (i + 1);
                dot.setAttribute('aria-label', dot.title);
                dot.style.cssText = [
                    'position:absolute',
                    'left:' + leftPct + '%',
                    'top:0',
                    'transform:translateX(-50%)',
                    'width:12px',
                    'height:12px',
                    'border-radius:50%',
                    'border:2px solid #fff',
                    'background:#2ca02c',
                    'box-shadow:0 0 0 2px rgba(0,0,0,0.20)',
                    'cursor:pointer',
                    'pointer-events:auto',
                    'padding:0'
                ].join(';');

                dot.addEventListener('click', function() {{
                    slider.value = String(idx);
                    updateTrack(idx);
                }});

                markerRail.appendChild(dot);
            }});
        }}

        let drawnLayers = [];
        let isInitialLoad = true;

        function clearLayers() {{
            drawnLayers.forEach(l => map.removeLayer(l));
            drawnLayers = [];
        }}

        function markerRole(p) {{
            if (p.after_break)  return 'Segment Start';
            if (p.before_break) return 'Segment End';
            return 'Waypoint';
        }}

        function formatSpeed(p) {{
            if (p.speed_kmh === null)
                return p.speed_knots === null ? 'N/A' : p.speed_knots + ' knots';
            return p.speed_kmh + ' km/h (' + (p.speed_knots ?? 'N/A') + ' knots)';
        }}

        function formatPopup(p) {{
            return '<b>' + p.time + '</b><br>'
                 + '<i>' + markerRole(p) + '</i><br>'
                 + 'Lat: ' + p.lat.toFixed(6) + '<br>'
                 + 'Lon: ' + p.lon.toFixed(6) + '<br>'
                 + 'Speed: ' + formatSpeed(p) + '<br>'
                 + 'Dist: ' + p.cum_km + ' km<br>'
                 + ALTITUDE_LABEL + ': ' + (p.altitude === null ? 'N/A' : p.altitude + ' m');
        }}

        const speedPalette = ['#0000ff','#00ffff','#00ff00','#ffff00','#ff8800','#ff0000'];
        const allSpeeds    = trackData.map(p => p.speed_kmh ?? 0);
        const minSpd       = Math.min(...allSpeeds);
        const maxSpd       = Math.max(...allSpeeds);

        function lerpColor(a, b, t) {{
            const ah = parseInt(a.replace('#',''), 16);
            const bh = parseInt(b.replace('#',''), 16);
            const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
            const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
            const rr = Math.round(ar + (br - ar) * t);
            const rg = Math.round(ag + (bg - ag) * t);
            const rb = Math.round(ab + (bb - ab) * t);
            return '#' + ((1 << 24) | (rr << 16) | (rg << 8) | rb).toString(16).slice(1);
        }}

        function speedToColor(spd) {{
            if (maxSpd === minSpd) return speedPalette[0];
            const t      = (spd - minSpd) / (maxSpd - minSpd);
            const scaled = t * (speedPalette.length - 1);
            const lo     = Math.floor(scaled);
            const hi     = Math.min(lo + 1, speedPalette.length - 1);
            return lerpColor(speedPalette[lo], speedPalette[hi], scaled - lo);
        }}

        function makePinIcon(color) {{
            const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="36" viewBox="0 0 24 36">
                <path d="M12 0C5.4 0 0 5.4 0 12c0 9 12 24 12 24s12-15 12-24C24 5.4 18.6 0 12 0z"
                      fill="${{color}}" stroke="#fff" stroke-width="1.5"/>
                <circle cx="12" cy="12" r="5" fill="#fff" opacity="0.85"/>
            </svg>`;
            return L.divIcon({{
                html:       svg,
                className:  '',
                iconSize:   [24, 36],
                iconAnchor: [12, 36],
                popupAnchor:[0, -36],
            }});
        }}

        function updateTrack(index) {{
            index = Math.max(0, Math.min(index, trackData.length - 1));
            clearLayers();

            const visited = trackData.slice(0, index + 1);   // drawn track
            const ahead   = trackData.slice(index + 1);      // not yet reached

            // ----------------------------------------------------------
            // 1. Shadow underline for visited segment
            // ----------------------------------------------------------
            let segCoords = [];
            function flushShadow() {{
                if (segCoords.length >= 2) {{
                    drawnLayers.push(
                        L.polyline(segCoords, {{
                            color:   '#333333',
                            weight:  POLYLINE_WEIGHT + 3,
                            opacity: 0.30,
                        }}).addTo(map)
                    );
                }}
                segCoords = [];
            }}
            visited.forEach(p => {{
                if (p.after_break && segCoords.length) flushShadow();
                segCoords.push([p.lat, p.lon]);
                if (p.before_break) flushShadow();
            }});
            flushShadow();

            // ----------------------------------------------------------
            // 2. Speed-coloured segments for visited points
            // ----------------------------------------------------------
            let segC = [], segS = [];
            function flushSeg() {{
                if (segC.length >= 2) {{
                    for (let j = 0; j < segC.length - 1; j++) {{
                        drawnLayers.push(
                            L.polyline([segC[j], segC[j + 1]], {{
                                color:   speedToColor(segS[j]),
                                weight:  POLYLINE_WEIGHT,
                                opacity: POLYLINE_OPACITY,
                            }}).addTo(map)
                        );
                    }}
                }}
                segC = []; segS = [];
            }}
            visited.forEach(p => {{
                if (p.after_break && segC.length) flushSeg();
                segC.push([p.lat, p.lon]);
                segS.push(p.speed_kmh ?? 0);
                if (p.before_break) flushSeg();
            }});
            flushSeg();

            // ----------------------------------------------------------
            // 3. Blue circles only for AHEAD (not-yet-visited) waypoints
            //    — gives the illusion the line "consumes" them
            // ----------------------------------------------------------
            ahead.forEach(p => {{
                if (p.after_break || p.before_break) return;
                drawnLayers.push(
                    L.circleMarker([p.lat, p.lon], {{
                        radius:      WAYPOINT_RADIUS,
                        color:       '#1a6fcc',
                        fill:        false,
                        opacity:     WAYPOINT_OPACITY,
                    }})
                    .bindPopup(formatPopup(p))
                    .addTo(map)
                );
            }});

            // ----------------------------------------------------------
            // 4. Boundary pin markers (all of them, visited + ahead)
            //    so start/end pins are always visible
            // ----------------------------------------------------------
            trackData.forEach(p => {{
                if (!p.after_break && !p.before_break) return;
                const color = p.after_break ? '#2ca02c' : '#d62728';
                drawnLayers.push(
                    L.marker([p.lat, p.lon], {{ icon: makePinIcon(color) }})
                    .bindPopup(formatPopup(p))
                    .addTo(map)
                );
            }});

            // ----------------------------------------------------------
            // 5. Current-position marker — orange, always on top
            // ----------------------------------------------------------
            const cur = trackData[index];
            drawnLayers.push(
                L.circleMarker([cur.lat, cur.lon], {{
                    radius:      13,
                    color:       '#ffffff',
                    weight:      4,
                    fill:        true,
                    fillColor:   '#ff6600',
                    fillOpacity: 0.95,
                    opacity:     1.0,
                }})
                .bindPopup(formatPopup(cur))
                .addTo(map)
            );

            // ----------------------------------------------------------
            // 6. Pan map to follow current point (no zoom change)
            // ----------------------------------------------------------
            if (!isInitialLoad) {{
                map.panTo([cur.lat, cur.lon], {{ animate: true, duration: 0.4 }});
            }}
            isInitialLoad = false;

            // ----------------------------------------------------------
            // 7. Update info bar
            // ----------------------------------------------------------
            timeLabel.textContent  = cur.time;
            distLabel.textContent  = 'Dist: ' + cur.cum_km + ' km';
            speedLabel.textContent = 'Speed: ' + formatSpeed(cur);
            altLabel.textContent   = ALTITUDE_LABEL + ': '
                                     + (cur.altitude === null ? 'N/A' : cur.altitude + ' m');
            roleLabel.textContent  = markerRole(cur);
        }}

        slider.addEventListener('input', e => updateTrack(parseInt(e.target.value, 10)));
        renderSegmentMarkers();
        updateTrack(parseInt(slider.value, 10));

        // Fit map to full track extent on load
        const allCoords = trackData.map(p => [p.lat, p.lon]);
        map.fitBounds(allCoords, {{ padding: [30, 30] }});
    }});
    </script>
    """



def _build_track_points(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for row in df.itertuples(index=False):
        records.append({
            "time":         row.time_str,
            "lat":          float(row.lat),
            "lon":          float(row.lon),
            "speed_knots":  None if pd.isna(row.speed_knots) else round(float(row.speed_knots), 1),
            "speed_kmh":    None if pd.isna(row.speed_kmh)   else round(float(row.speed_kmh), 1),
            "altitude":     None if pd.isna(row.altitude)     else round(float(row.altitude), 1),
            "cum_km":       round(float(row.cum_km), 3),
            "after_break":  bool(row.after_break),
            "before_break": bool(row.before_break),
        })
    return records


# ---------------------------------------------------------------------------
# Overlay injection
# ---------------------------------------------------------------------------
def add_overlays(
    m: folium.Map,
    df_display: pd.DataFrame,
    total_km: float,
    track_points: list[dict],
) -> None:
    m.get_root().html.add_child(folium.Element(_legend_html(total_km)))
    m.get_root().html.add_child(folium.Element(_slider_html(len(df_display))))
    m.get_root().html.add_child(folium.Element(_script_html(track_points, m.get_name())))


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def parse_and_visualize(file_path: Path, overwrite: bool = False) -> None:
    df = load_gps_csv(file_path)
    df_display = sample_track(df, DISPLAY_STEP)
    df_display["cum_km"], total_km = compute_thresholded_distance(df_display)

    m, _ = build_base_map(df_display)
    track_points = _build_track_points(df_display)
    add_overlays(m, df_display, total_km, track_points)

    output_path = file_path.with_name(f"{file_path.stem}_map.html")
    if output_path.exists() and not overwrite:
        print(f"Skipped (already exists): {output_path}")
        return

    m.save(output_path)
    print(f"Map saved to: {output_path}  (total distance: {total_km:.3f} km)")


def process_input(path: str, overwrite: bool = False) -> None:
    p = Path(path)
    if p.is_dir():
        dat_files = sorted(p.glob("*.dat"))
        if not dat_files:
            print(f"No .dat files found in directory: {path}")
            return
        for idx, dat_file in enumerate(dat_files, 1):
            print(f"[{idx}/{len(dat_files)}] Processing: {dat_file}")
            try:
                parse_and_visualize(dat_file, overwrite=overwrite)
            except Exception as e:
                print(f"  Error: {e}")
    elif p.is_file():
        parse_and_visualize(p, overwrite=overwrite)
    else:
        print(f"Error: '{path}' is not a valid file or directory.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPS track visualiser")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing maps")
    args = parser.parse_args()
    process_input(args.path, overwrite=args.overwrite)
