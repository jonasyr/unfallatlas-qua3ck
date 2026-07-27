"""Risk Predictor page: interactive KSI-risk prediction form."""

import folium
import streamlit as st
from streamlit_folium import st_folium

from unfallatlas.viz.streamlit_app import (
    DEFAULT_WIDGET_VALUES,
    EXAMPLE_HIGH_RISK,
    EXAMPLE_LOW_RISK,
    LICHTVERH_LABELS,
    LIMITATIONS_TEXT,
    SEVERITY_COLORS,
    STRZUSTAND_LABELS,
    UART_LABELS,
    UTYP1_LABELS,
    WEEKDAY_LABELS,
    build_input_row,
    get_column_spec,
    load_categorical_options,
    load_champion_model,
    load_inference_contract,
    nearest_location_features,
    predict_ksi,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_scenario(scenario: dict) -> None:
    """Push every field of an example scenario into the widget session state."""
    st.session_state["picked_lat"] = scenario["LAT"]
    st.session_state["picked_lon"] = scenario["LON"]
    st.session_state["picked_uregbez"] = scenario["UREGBEZ"]
    st.session_state["picked_ukreis"] = scenario["UKREIS"]
    st.session_state["risk_umonat"] = scenario["UMONAT"]
    st.session_state["risk_ustunde"] = scenario["USTUNDE"]
    st.session_state["risk_uwochentag"] = scenario["UWOCHENTAG"]
    st.session_state["risk_uart"] = scenario["UART"]
    st.session_state["risk_utyp1"] = scenario["UTYP1"]
    st.session_state["risk_ulichtverh"] = scenario["ULICHTVERH"]
    st.session_state["risk_strzustand"] = scenario["STRZUSTAND"]
    st.session_state["risk_ist_rad"] = scenario["IstRad"]
    st.session_state["risk_ist_pkw"] = scenario["IstPKW"]
    st.session_state["risk_ist_fuss"] = scenario["IstFuss"]
    st.session_state["risk_ist_krad"] = scenario["IstKrad"]
    st.session_state["risk_ist_gkfz"] = scenario["IstGkfz"]
    st.session_state["risk_ist_sonstig"] = scenario["IstSonstig"]
    st.session_state["risk_dwd_station_id"] = scenario["dwd_station_id"]
    st.session_state["risk_dwd_station_dist_km"] = scenario["dwd_station_dist_km"]
    st.session_state["risk_dwd_temp"] = scenario["dwd_temp_air_2m"]
    st.session_state["risk_dwd_precip"] = scenario["dwd_precip_mm"]
    st.session_state["risk_dwd_visibility"] = scenario["dwd_visibility_m"]
    st.session_state["risk_dwd_wind"] = scenario["dwd_wind_speed_ms"]
    st.session_state["risk_h3_cell"] = scenario["h3_cell"]
    st.session_state["risk_osm_road_class"] = scenario["osm_dominant_road_class"]
    st.session_state["risk_osm_maxspeed_mean"] = scenario["osm_maxspeed_mean"]
    st.session_state["risk_osm_maxspeed_max"] = scenario["osm_maxspeed_max"]
    st.session_state["risk_osm_road_density"] = scenario["osm_road_density"]
    st.session_state["risk_osm_way_count"] = scenario["osm_way_count"]


st.title("Risk Predictor")
st.caption(
    "Estimate the probability that a described accident is KSI "
    "(killed or seriously injured) vs. slight."
)

contract = load_inference_contract()
defaults = DEFAULT_WIDGET_VALUES
ukreis_options = load_categorical_options("UKREIS")
uregbez_categories = get_column_spec(contract, "UREGBEZ")["categories"]
road_class_categories = get_column_spec(contract, "osm_dominant_road_class")["categories"]

lon_spec = get_column_spec(contract, "LON")
lat_spec = get_column_spec(contract, "LAT")
temp_spec = get_column_spec(contract, "dwd_temp_air_2m")
precip_spec = get_column_spec(contract, "dwd_precip_mm")
vis_spec = get_column_spec(contract, "dwd_visibility_m")
wind_spec = get_column_spec(contract, "dwd_wind_speed_ms")
station_dist_spec = get_column_spec(contract, "dwd_station_dist_km")
maxspeed_mean_spec = get_column_spec(contract, "osm_maxspeed_mean")
maxspeed_max_spec = get_column_spec(contract, "osm_maxspeed_max")
density_spec = get_column_spec(contract, "osm_road_density")
way_count_spec = get_column_spec(contract, "osm_way_count")

for key, value in {
    "picked_lat": defaults["LAT"],
    "picked_lon": defaults["LON"],
    "picked_uregbez": defaults["UREGBEZ"],
    "picked_ukreis": defaults["UKREIS"],
    "last_processed_click": None,
    "risk_umonat": defaults["UMONAT"],
    "risk_ustunde": defaults["USTUNDE"],
    "risk_uwochentag": defaults["UWOCHENTAG"],
    "risk_uart": defaults["UART"],
    "risk_utyp1": defaults["UTYP1"],
    "risk_ulichtverh": defaults["ULICHTVERH"],
    "risk_strzustand": defaults["STRZUSTAND"],
    "risk_ist_rad": defaults["IstRad"],
    "risk_ist_pkw": defaults["IstPKW"],
    "risk_ist_fuss": defaults["IstFuss"],
    "risk_ist_krad": defaults["IstKrad"],
    "risk_ist_gkfz": defaults["IstGkfz"],
    "risk_ist_sonstig": defaults["IstSonstig"],
    "risk_dwd_station_id": defaults["dwd_station_id"],
    "risk_dwd_station_dist_km": defaults["dwd_station_dist_km"],
    "risk_dwd_temp": defaults["dwd_temp_air_2m"],
    "risk_dwd_precip": defaults["dwd_precip_mm"],
    "risk_dwd_visibility": defaults["dwd_visibility_m"],
    "risk_dwd_wind": defaults["dwd_wind_speed_ms"],
    "risk_h3_cell": defaults["h3_cell"],
    "risk_osm_road_class": defaults["osm_dominant_road_class"],
    "risk_osm_maxspeed_mean": defaults["osm_maxspeed_mean"],
    "risk_osm_maxspeed_max": defaults["osm_maxspeed_max"],
    "risk_osm_road_density": defaults["osm_road_density"],
    "risk_osm_way_count": defaults["osm_way_count"],
}.items():
    st.session_state.setdefault(key, value)

st.subheader("Try an example")
ex1, ex2 = st.columns(2)
ex1.button(
    "Load a clearly-not-KSI example",
    use_container_width=True,
    on_click=_load_scenario,
    args=(EXAMPLE_LOW_RISK,),
)
ex2.button(
    "Load a clearly-KSI example",
    use_container_width=True,
    on_click=_load_scenario,
    args=(EXAMPLE_HIGH_RISK,),
)
st.caption(
    "Real accident records the champion model scores near the extremes on - "
    "useful because typical hand-picked inputs tend to land close to the "
    "50/50 decision threshold instead."
)

st.subheader("Pick a location")
st.caption(
    "Click a point on the map to set the accident's longitude/latitude and "
    "auto-fill the administrative codes and road/weather context below from "
    "the nearest recorded accident. Clicks outside Germany's covered area are "
    "ignored with a warning below. The map reloads after every click, which "
    "can take a moment - that's expected, not a freeze."
)
picker_map = folium.Map(
    location=[st.session_state["picked_lat"], st.session_state["picked_lon"]],
    zoom_start=6,
    tiles="cartodbdark_matter",
)
folium.Marker(
    [st.session_state["picked_lat"], st.session_state["picked_lon"]], tooltip="Selected location"
).add_to(picker_map)
map_state = st_folium(
    picker_map,
    height=350,
    width=None,
    key="location_picker",
    returned_objects=["last_clicked"],
)

if map_state and map_state.get("last_clicked"):
    clicked = (map_state["last_clicked"]["lat"], map_state["last_clicked"]["lng"])
    # Only act the first time a given click is seen - st_folium keeps replaying
    # the same last_clicked value on every later rerun (e.g. form submission),
    # which would otherwise re-show a stale out-of-bounds warning forever.
    if clicked != st.session_state["last_processed_click"]:
        st.session_state["last_processed_click"] = clicked
        clicked_lat, clicked_lon = clicked
        if (
            lat_spec["min"] <= clicked_lat <= lat_spec["max"]
            and lon_spec["min"] <= clicked_lon <= lon_spec["max"]
        ):
            st.session_state["picked_lat"] = clicked_lat
            st.session_state["picked_lon"] = clicked_lon
            features = nearest_location_features(clicked_lat, clicked_lon)
            if features["UREGBEZ"] in uregbez_categories:
                st.session_state["picked_uregbez"] = features["UREGBEZ"]
            if features["UKREIS"] in ukreis_options:
                st.session_state["picked_ukreis"] = features["UKREIS"]
            if features["osm_dominant_road_class"] in road_class_categories:
                st.session_state["risk_osm_road_class"] = features["osm_dominant_road_class"]
            st.session_state["risk_dwd_station_id"] = features["dwd_station_id"]
            st.session_state["risk_h3_cell"] = features["h3_cell"]
            # The nearest record can have NaN in some optional columns (not
            # every OSM way carries a maxspeed tag, not every accident has a
            # nearby weather reading) - skip those fields rather than
            # crashing _clamp() on a None, leaving the previous value in place.
            for key, spec_name, spec in (
                ("risk_dwd_station_dist_km", "dwd_station_dist_km", station_dist_spec),
                ("risk_dwd_temp", "dwd_temp_air_2m", temp_spec),
                ("risk_dwd_precip", "dwd_precip_mm", precip_spec),
                ("risk_dwd_visibility", "dwd_visibility_m", vis_spec),
                ("risk_dwd_wind", "dwd_wind_speed_ms", wind_spec),
                ("risk_osm_maxspeed_mean", "osm_maxspeed_mean", maxspeed_mean_spec),
                ("risk_osm_maxspeed_max", "osm_maxspeed_max", maxspeed_max_spec),
                ("risk_osm_road_density", "osm_road_density", density_spec),
                ("risk_osm_way_count", "osm_way_count", way_count_spec),
            ):
                raw_value = features[spec_name]
                if raw_value is not None:
                    st.session_state[key] = _clamp(raw_value, spec["min"], spec["max"])
        else:
            st.warning(
                f"Clicked point ({clicked_lat:.4f}, {clicked_lon:.4f}) is outside the "
                f"covered range (lat {lat_spec['min']:.2f}-{lat_spec['max']:.2f}, "
                f"lon {lon_spec['min']:.2f}-{lon_spec['max']:.2f}) and was ignored."
            )

st.caption(
    f"Selected: lat {st.session_state['picked_lat']:.4f}, lon {st.session_state['picked_lon']:.4f}"
)

with st.form("risk_predictor_form"):
    st.subheader("When, and administrative area")
    c1, c2, c3 = st.columns(3)
    with c1:
        uregbez = st.selectbox(
            "Regierungsbezirk code (UREGBEZ)",
            options=uregbez_categories,
            key="picked_uregbez",
        )
        ukreis = st.selectbox(
            "Kreis code (UKREIS)",
            options=ukreis_options,
            key="picked_ukreis",
        )
        st.caption(
            "Auto-filled from the nearest recorded accident to your map click "
            "above; edit manually if needed. These stay dataset-internal codes: "
            "this dataset has no ULAND (Bundesland) column at all, so there is "
            "no way to resolve them to official Gemeindeschlüssel/region names."
        )
    with c2:
        umonat_spec = get_column_spec(contract, "UMONAT")
        umonat = st.slider(
            "Month (UMONAT)",
            min_value=int(umonat_spec["min"]),
            max_value=int(umonat_spec["max"]),
            key="risk_umonat",
        )
        ustunde_spec = get_column_spec(contract, "USTUNDE")
        ustunde = st.slider(
            "Hour (USTUNDE)",
            min_value=int(ustunde_spec["min"]),
            max_value=int(ustunde_spec["max"]),
            key="risk_ustunde",
        )
    with c3:
        uwochentag = st.selectbox(
            "Weekday (UWOCHENTAG)",
            options=list(WEEKDAY_LABELS.keys()),
            format_func=lambda code: WEEKDAY_LABELS[code],
            key="risk_uwochentag",
        )

    st.subheader("Accident characteristics")
    c4, c5, c6 = st.columns(3)
    with c4:
        uart_spec = get_column_spec(contract, "UART")
        uart_options = list(range(int(uart_spec["min"]), int(uart_spec["max"]) + 1))
        uart = st.selectbox(
            "Accident type (UART)",
            options=uart_options,
            format_func=lambda code: UART_LABELS.get(code, str(code)),
            key="risk_uart",
        )
        utyp1_spec = get_column_spec(contract, "UTYP1")
        utyp1_options = list(range(int(utyp1_spec["min"]), int(utyp1_spec["max"]) + 1))
        utyp1 = st.selectbox(
            "Accident category (UTYP1)",
            options=utyp1_options,
            format_func=lambda code: UTYP1_LABELS.get(code, str(code)),
            key="risk_utyp1",
        )
    with c5:
        ulichtverh = st.selectbox(
            "Light conditions (ULICHTVERH)",
            options=list(LICHTVERH_LABELS.keys()),
            format_func=lambda code: LICHTVERH_LABELS[code],
            key="risk_ulichtverh",
        )
        strzustand = st.selectbox(
            "Road condition (STRZUSTAND)",
            options=list(STRZUSTAND_LABELS.keys()),
            format_func=lambda code: STRZUSTAND_LABELS[code],
            key="risk_strzustand",
        )
    with c6:
        ist_rad = st.checkbox("Cyclist involved (IstRad)", key="risk_ist_rad")
        ist_pkw = st.checkbox("Car involved (IstPKW)", key="risk_ist_pkw")
        ist_fuss = st.checkbox("Pedestrian involved (IstFuss)", key="risk_ist_fuss")
        ist_krad = st.checkbox("Motorcycle involved (IstKrad)", key="risk_ist_krad")
        ist_gkfz = st.checkbox("Heavy goods vehicle involved (IstGkfz)", key="risk_ist_gkfz")
        ist_sonstig = st.checkbox("Other vehicle involved (IstSonstig)", key="risk_ist_sonstig")

    st.subheader("Weather")
    st.caption("Auto-filled from the nearest recorded accident to your map click above.")
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        dwd_temp_air_2m = st.slider(
            "Air temperature (C)",
            min_value=float(temp_spec["min"]),
            max_value=float(temp_spec["max"]),
            key="risk_dwd_temp",
        )
    with c10:
        dwd_precip_mm = st.slider(
            "Precipitation (mm)",
            min_value=float(precip_spec["min"]),
            max_value=float(precip_spec["max"]),
            key="risk_dwd_precip",
        )
    with c11:
        dwd_visibility_m = st.slider(
            "Visibility (m)",
            min_value=float(vis_spec["min"]),
            max_value=float(vis_spec["max"]),
            key="risk_dwd_visibility",
        )
    with c12:
        dwd_wind_speed_ms = st.slider(
            "Wind speed (m/s)",
            min_value=float(wind_spec["min"]),
            max_value=float(wind_spec["max"]),
            key="risk_dwd_wind",
        )
    st.subheader("Road context (OpenStreetMap)")
    st.caption("Auto-filled from the nearest recorded accident to your map click above.")
    c13, c14, c15, c16 = st.columns(4)
    with c13:
        osm_dominant_road_class = st.selectbox(
            "Dominant road class",
            options=road_class_categories,
            key="risk_osm_road_class",
        )
    with c14:
        osm_maxspeed_mean = st.slider(
            "Mean speed limit (km/h)",
            min_value=float(maxspeed_mean_spec["min"]),
            max_value=float(maxspeed_mean_spec["max"]),
            key="risk_osm_maxspeed_mean",
        )
    with c15:
        osm_maxspeed_max = st.slider(
            "Max speed limit (km/h)",
            min_value=float(maxspeed_max_spec["min"]),
            max_value=float(maxspeed_max_spec["max"]),
            key="risk_osm_maxspeed_max",
        )
    with c16:
        osm_road_density = st.slider(
            "Road density (H3 cell)",
            min_value=float(density_spec["min"]),
            max_value=float(density_spec["max"]),
            key="risk_osm_road_density",
        )
        osm_way_count = st.slider(
            "Road way count (H3 cell)",
            min_value=float(way_count_spec["min"]),
            max_value=float(way_count_spec["max"]),
            key="risk_osm_way_count",
        )

    submitted = st.form_submit_button("Predict KSI risk")

if submitted:
    precip_bucket_categories = get_column_spec(contract, "_precip_bucket")["categories"]
    dry_bucket = next(c for c in precip_bucket_categories if c.startswith("dry"))
    light_bucket = next(c for c in precip_bucket_categories if c.startswith("light"))
    precip_bucket = dry_bucket if dwd_precip_mm == 0 else light_bucket

    widget_values = {
        "UREGBEZ": uregbez,
        "UKREIS": ukreis,
        "UMONAT": umonat,
        "USTUNDE": ustunde,
        "UWOCHENTAG": uwochentag,
        "UART": uart,
        "UTYP1": utyp1,
        "ULICHTVERH": ulichtverh,
        "STRZUSTAND": strzustand,
        "IstRad": ist_rad,
        "IstPKW": ist_pkw,
        "IstFuss": ist_fuss,
        "IstKrad": ist_krad,
        "IstGkfz": ist_gkfz,
        "IstSonstig": ist_sonstig,
        "LON": st.session_state["picked_lon"],
        "LAT": st.session_state["picked_lat"],
        "dwd_station_id": st.session_state["risk_dwd_station_id"],
        "dwd_station_dist_km": st.session_state["risk_dwd_station_dist_km"],
        "dwd_temp_air_2m": dwd_temp_air_2m,
        "dwd_precip_mm": dwd_precip_mm,
        "dwd_visibility_m": dwd_visibility_m,
        "dwd_wind_speed_ms": dwd_wind_speed_ms,
        "_precip_bucket": precip_bucket,
        "h3_cell": st.session_state["risk_h3_cell"],
        "osm_dominant_road_class": osm_dominant_road_class,
        "osm_maxspeed_mean": osm_maxspeed_mean,
        "osm_maxspeed_max": osm_maxspeed_max,
        "osm_road_density": osm_road_density,
        "osm_way_count": osm_way_count,
    }

    try:
        model = load_champion_model()
        row = build_input_row(widget_values, contract)
        proba, prediction = predict_ksi(model, row, contract["threshold"])
    except KeyError as exc:
        st.error(f"Could not build the model input row: {exc}")
        st.stop()

    st.session_state["last_prediction"] = {
        "inputs": widget_values,
        "proba": proba,
        "prediction": prediction,
    }

    label = "KSI (killed or seriously injured)" if prediction == 1 else "Slight injury"
    color = SEVERITY_COLORS["KSI"] if prediction == 1 else SEVERITY_COLORS["slight"]
    st.markdown(
        f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True
    )
    st.metric(
        "KSI probability", f"{proba:.1%}", help=f"Decision threshold: {contract['threshold']:.1%}"
    )

with st.expander("Limitations"):
    st.markdown(LIMITATIONS_TEXT)
