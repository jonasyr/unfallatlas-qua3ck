"""Risk Predictor page: interactive KSI-risk prediction form."""

import folium
import streamlit as st
from streamlit_folium import st_folium

from unfallatlas.viz.streamlit_app import (
    DEFAULT_WIDGET_VALUES,
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
    predict_ksi,
)

st.title("Risk Predictor")
st.caption(
    "Estimate the probability that a described accident is KSI "
    "(killed or seriously injured) vs. slight."
)

contract = load_inference_contract()
defaults = DEFAULT_WIDGET_VALUES
ukreis_options = load_categorical_options("UKREIS")

lon_spec = get_column_spec(contract, "LON")
lat_spec = get_column_spec(contract, "LAT")
st.session_state.setdefault("picked_lat", defaults["LAT"])
st.session_state.setdefault("picked_lon", defaults["LON"])

st.subheader("Pick a location")
st.caption(
    "Click a point on the map to set the accident's longitude/latitude. "
    "Clicks outside Germany's covered area are ignored with a warning below."
)
picker_map = folium.Map(
    location=[st.session_state["picked_lat"], st.session_state["picked_lon"]], zoom_start=6
)
folium.Marker(
    [st.session_state["picked_lat"], st.session_state["picked_lon"]], tooltip="Selected location"
).add_to(picker_map)
map_state = st_folium(picker_map, height=350, width=None, key="location_picker")

if map_state and map_state.get("last_clicked"):
    clicked_lat = map_state["last_clicked"]["lat"]
    clicked_lon = map_state["last_clicked"]["lng"]
    if (
        lat_spec["min"] <= clicked_lat <= lat_spec["max"]
        and lon_spec["min"] <= clicked_lon <= lon_spec["max"]
    ):
        st.session_state["picked_lat"] = clicked_lat
        st.session_state["picked_lon"] = clicked_lon
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
    st.subheader("When and where")
    c1, c2, c3 = st.columns(3)
    with c1:
        uregbez_categories = get_column_spec(contract, "UREGBEZ")["categories"]
        uregbez = st.selectbox(
            "Regierungsbezirk code (UREGBEZ)",
            options=uregbez_categories,
            index=uregbez_categories.index(defaults["UREGBEZ"]),
        )
        ukreis = st.selectbox(
            "Kreis code (UKREIS)",
            options=ukreis_options,
            index=ukreis_options.index(defaults["UKREIS"]),
        )
        st.caption(
            "Dataset-internal codes, not official region names: the model's "
            "feature set doesn't include the Bundesland key (ULAND) needed to "
            "resolve these to an official Gemeindeschlüssel/region name."
        )
    with c2:
        umonat_spec = get_column_spec(contract, "UMONAT")
        umonat = st.slider(
            "Month (UMONAT)",
            min_value=int(umonat_spec["min"]),
            max_value=int(umonat_spec["max"]),
            value=defaults["UMONAT"],
        )
        ustunde_spec = get_column_spec(contract, "USTUNDE")
        ustunde = st.slider(
            "Hour (USTUNDE)",
            min_value=int(ustunde_spec["min"]),
            max_value=int(ustunde_spec["max"]),
            value=defaults["USTUNDE"],
        )
    with c3:
        uwochentag_label = st.selectbox(
            "Weekday (UWOCHENTAG)",
            options=list(WEEKDAY_LABELS.values()),
            index=defaults["UWOCHENTAG"] - 1,
        )
        uwochentag = {v: k for k, v in WEEKDAY_LABELS.items()}[uwochentag_label]

    st.subheader("Accident characteristics")
    c4, c5, c6 = st.columns(3)
    with c4:
        uart_spec = get_column_spec(contract, "UART")
        uart_options = list(range(int(uart_spec["min"]), int(uart_spec["max"]) + 1))
        uart = st.selectbox(
            "Accident type (UART)",
            options=uart_options,
            index=uart_options.index(defaults["UART"]),
            format_func=lambda code: UART_LABELS.get(code, str(code)),
        )
        utyp1_spec = get_column_spec(contract, "UTYP1")
        utyp1_options = list(range(int(utyp1_spec["min"]), int(utyp1_spec["max"]) + 1))
        utyp1 = st.selectbox(
            "Accident category (UTYP1)",
            options=utyp1_options,
            index=utyp1_options.index(defaults["UTYP1"]),
            format_func=lambda code: UTYP1_LABELS.get(code, str(code)),
        )
    with c5:
        ulichtverh_label = st.selectbox(
            "Light conditions (ULICHTVERH)",
            options=list(LICHTVERH_LABELS.values()),
            index=defaults["ULICHTVERH"],
        )
        ulichtverh = {v: k for k, v in LICHTVERH_LABELS.items()}[ulichtverh_label]
        strzustand_label = st.selectbox(
            "Road condition (STRZUSTAND)",
            options=list(STRZUSTAND_LABELS.values()),
            index=defaults["STRZUSTAND"],
        )
        strzustand = {v: k for k, v in STRZUSTAND_LABELS.items()}[strzustand_label]
    with c6:
        ist_rad = st.checkbox("Cyclist involved (IstRad)", value=defaults["IstRad"])
        ist_pkw = st.checkbox("Car involved (IstPKW)", value=defaults["IstPKW"])
        ist_fuss = st.checkbox("Pedestrian involved (IstFuss)", value=defaults["IstFuss"])
        ist_krad = st.checkbox("Motorcycle involved (IstKrad)", value=defaults["IstKrad"])
        ist_gkfz = st.checkbox("Heavy goods vehicle involved (IstGkfz)", value=defaults["IstGkfz"])
        ist_sonstig = st.checkbox(
            "Other vehicle involved (IstSonstig)", value=defaults["IstSonstig"]
        )

    st.subheader("Weather")
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        temp_spec = get_column_spec(contract, "dwd_temp_air_2m")
        dwd_temp_air_2m = st.slider(
            "Air temperature (C)",
            min_value=float(temp_spec["min"]),
            max_value=float(temp_spec["max"]),
            value=defaults["dwd_temp_air_2m"],
        )
    with c10:
        precip_spec = get_column_spec(contract, "dwd_precip_mm")
        dwd_precip_mm = st.slider(
            "Precipitation (mm)",
            min_value=float(precip_spec["min"]),
            max_value=float(precip_spec["max"]),
            value=defaults["dwd_precip_mm"],
        )
    with c11:
        vis_spec = get_column_spec(contract, "dwd_visibility_m")
        dwd_visibility_m = st.slider(
            "Visibility (m)",
            min_value=float(vis_spec["min"]),
            max_value=float(vis_spec["max"]),
            value=defaults["dwd_visibility_m"],
        )
    with c12:
        wind_spec = get_column_spec(contract, "dwd_wind_speed_ms")
        dwd_wind_speed_ms = st.slider(
            "Wind speed (m/s)",
            min_value=float(wind_spec["min"]),
            max_value=float(wind_spec["max"]),
            value=defaults["dwd_wind_speed_ms"],
        )
    st.subheader("Road context (OpenStreetMap)")
    c13, c14, c15, c16 = st.columns(4)
    with c13:
        road_class_categories = get_column_spec(contract, "osm_dominant_road_class")["categories"]
        osm_dominant_road_class = st.selectbox(
            "Dominant road class",
            options=road_class_categories,
            index=road_class_categories.index(defaults["osm_dominant_road_class"]),
        )
    with c14:
        maxspeed_mean_spec = get_column_spec(contract, "osm_maxspeed_mean")
        osm_maxspeed_mean = st.slider(
            "Mean speed limit (km/h)",
            min_value=float(maxspeed_mean_spec["min"]),
            max_value=float(maxspeed_mean_spec["max"]),
            value=defaults["osm_maxspeed_mean"],
        )
    with c15:
        maxspeed_max_spec = get_column_spec(contract, "osm_maxspeed_max")
        osm_maxspeed_max = st.slider(
            "Max speed limit (km/h)",
            min_value=float(maxspeed_max_spec["min"]),
            max_value=float(maxspeed_max_spec["max"]),
            value=defaults["osm_maxspeed_max"],
        )
    with c16:
        density_spec = get_column_spec(contract, "osm_road_density")
        osm_road_density = st.slider(
            "Road density (H3 cell)",
            min_value=float(density_spec["min"]),
            max_value=float(density_spec["max"]),
            value=defaults["osm_road_density"],
        )
        way_count_spec = get_column_spec(contract, "osm_way_count")
        osm_way_count = st.slider(
            "Road way count (H3 cell)",
            min_value=float(way_count_spec["min"]),
            max_value=float(way_count_spec["max"]),
            value=defaults["osm_way_count"],
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
        "dwd_station_id": defaults["dwd_station_id"],
        "dwd_station_dist_km": defaults["dwd_station_dist_km"],
        "dwd_temp_air_2m": dwd_temp_air_2m,
        "dwd_precip_mm": dwd_precip_mm,
        "dwd_visibility_m": dwd_visibility_m,
        "dwd_wind_speed_ms": dwd_wind_speed_ms,
        "_precip_bucket": precip_bucket,
        "h3_cell": defaults["h3_cell"],
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
