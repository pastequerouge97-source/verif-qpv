#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Streamlit pour la vérification QPV.

Lancement :
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import tempfile
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from verif_qpv import (
    QPV_CODE_CANDIDATES,
    QPV_NAME_CANDIDATES,
    QPV_DATASET_SLUG,
    detect_col,
    download_qpv_dataset,
    find_qpv,
    find_qpv_batch,
    geocode,
    geocode_batch_csv,
    load_qpv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration générale
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vérification QPV",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Vérification QPV")
st.caption(
    "Géocodage via l'API BAN + test point-dans-polygone sur le référentiel QPV 2024 "
    "de l'ANCT. Les adresses transitent uniquement par l'API publique "
    "adresse.data.gouv.fr."
)

CACHE_DIR = Path.home() / ".cache" / "verif-qpv"


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du référentiel QPV (avec cache)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du référentiel QPV…")
def _load_qpv_from_path(path: str):
    gdf = load_qpv(path)
    sindex = gdf.sindex
    code_col = detect_col(gdf, QPV_CODE_CANDIDATES)
    name_col = detect_col(gdf, QPV_NAME_CANDIDATES)
    return gdf, sindex, code_col, name_col


@st.cache_resource(show_spinner="Chargement du référentiel QPV…")
def _load_qpv_from_bytes(content_hash: str, content: bytes, filename: str):
    tmpdir = Path(tempfile.mkdtemp(prefix="qpv_streamlit_"))
    target = tmpdir / filename
    target.write_bytes(content)
    gdf = load_qpv(str(target))
    sindex = gdf.sindex
    code_col = detect_col(gdf, QPV_CODE_CANDIDATES)
    name_col = detect_col(gdf, QPV_NAME_CANDIDATES)
    return gdf, sindex, code_col, name_col


def _bytes_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — sélection du référentiel QPV (3 modes)
# ─────────────────────────────────────────────────────────────────────────────

qpv_gdf = None
sindex = None
code_col = name_col = None

with st.sidebar:
    st.header("⚙️ Référentiel QPV")

    mode = st.radio(
        "Comment fournir le référentiel ?",
        ("📥 Télécharger depuis data.gouv.fr", "📂 Uploader un fichier", "📁 Chemin local"),
        index=0,
    )

    if mode == "📥 Télécharger depuis data.gouv.fr":
        st.caption(
            f"Source : [data.gouv.fr]"
            f"(https://www.data.gouv.fr/datasets/{QPV_DATASET_SLUG})"
        )
        cached_files = sorted(CACHE_DIR.glob("*.zip")) if CACHE_DIR.exists() else []
        if cached_files:
            st.success(f"✓ Cache local : `{cached_files[0].name}`")

        if st.button("⬇ Télécharger / charger le QPV", use_container_width=True):
            progress = st.progress(0.0, text="Téléchargement…")
            try:
                def _cb(done, total):
                    if total:
                        progress.progress(min(done / total, 1.0), text=f"{done/1e6:.1f} / {total/1e6:.1f} Mo")
                zip_path = download_qpv_dataset(CACHE_DIR, progress_cb=_cb)
                progress.empty()
                qpv_gdf, sindex, code_col, name_col = _load_qpv_from_path(str(zip_path))
                st.session_state["qpv_path"] = str(zip_path)
                st.session_state["qpv_loaded_via"] = "download"
            except Exception as e:
                st.error(f"Échec du téléchargement : {e}")

        # Si on a déjà chargé dans la session, on garde
        if "qpv_path" in st.session_state and st.session_state.get("qpv_loaded_via") == "download":
            try:
                qpv_gdf, sindex, code_col, name_col = _load_qpv_from_path(
                    st.session_state["qpv_path"]
                )
            except Exception:
                pass

    elif mode == "📂 Uploader un fichier":
        st.caption(
            "Accepte `.zip` (shapefile zippé, recommandé), `.geojson`, `.gpkg`. "
            "Télécharge le shapefile depuis "
            f"[data.gouv.fr](https://www.data.gouv.fr/datasets/{QPV_DATASET_SLUG})."
        )
        up = st.file_uploader(
            "Fichier QPV",
            type=["zip", "geojson", "json", "gpkg"],
            label_visibility="collapsed",
        )
        if up is not None:
            try:
                content = up.getvalue()
                qpv_gdf, sindex, code_col, name_col = _load_qpv_from_bytes(
                    _bytes_hash(content), content, up.name
                )
                st.session_state["qpv_loaded_via"] = "upload"
            except Exception as e:
                st.error(f"Erreur de chargement : {e}")

    else:  # chemin local
        st.caption("Chemin **local** vers un fichier QPV (`.shp`, `.geojson`, `.gpkg`, `.zip`).")
        default_path = st.session_state.get("qpv_path_local", "")
        qpv_path = st.text_input(
            "Chemin",
            value=default_path,
            placeholder="C:/data/qpv/QP2024_France_hexagonale.shp",
            label_visibility="collapsed",
        )
        if qpv_path:
            if not Path(qpv_path).exists():
                st.error("Fichier introuvable à ce chemin.")
            else:
                try:
                    qpv_gdf, sindex, code_col, name_col = _load_qpv_from_path(qpv_path)
                    st.session_state["qpv_path_local"] = qpv_path
                    st.session_state["qpv_loaded_via"] = "local"
                except Exception as e:
                    st.error(f"Erreur de chargement : {e}")

    qpv_loaded = qpv_gdf is not None
    if qpv_loaded:
        st.success(f"✓ {len(qpv_gdf)} polygones QPV chargés")
        st.caption(f"Colonne code : `{code_col or '—'}`  •  Colonne nom : `{name_col or '—'}`")
    else:
        st.info("⬇ Charge le référentiel pour activer la vérification.")

    st.divider()
    st.markdown(
        "**Source officielle**  \n"
        f"[Référentiel QPV — data.gouv.fr](https://www.data.gouv.fr/datasets/{QPV_DATASET_SLUG})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers UI
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _geocode_cached(query: str, postcode: str, citycode: str) -> dict:
    """Cache de géocodage unitaire (TTL 24h). La session HTTP n'est pas mise en cache."""
    return geocode(query, postcode=postcode, citycode=citycode)


def fmt_score(s: Optional[float]) -> str:
    if s is None or pd.isna(s):
        return "—"
    pct = int(round(float(s) * 100))
    badge = "🟢" if s >= 0.8 else ("🟡" if s >= 0.5 else "🔴")
    return f"{badge} {pct} %"


def _qpv_geojson_subset(gdf, lat: float, lon: float, radius_deg: float = 0.05):
    """Renvoie un GeoJSON des polygones autour d'un point pour affichage."""
    bbox = (lon - radius_deg, lat - radius_deg, lon + radius_deg, lat + radius_deg)
    sub = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return sub.__geo_interface__ if len(sub) else None


def _qpv_geojson_for_points(gdf, points: pd.DataFrame, radius_deg: float = 0.05):
    """Renvoie un GeoJSON des polygones intersectant l'enveloppe des points."""
    if points.empty:
        return None
    lat_min, lat_max = points["lat"].min() - radius_deg, points["lat"].max() + radius_deg
    lon_min, lon_max = points["lon"].min() - radius_deg, points["lon"].max() + radius_deg
    sub = gdf.cx[lon_min:lon_max, lat_min:lat_max]
    return sub.__geo_interface__ if len(sub) else None


# ─────────────────────────────────────────────────────────────────────────────
# Onglets
# ─────────────────────────────────────────────────────────────────────────────

tab_unitaire, tab_lot = st.tabs(["🔍 Adresse unitaire", "📂 Lot CSV"])


# ── Onglet 1 : adresse unitaire ─────────────────────────────────────────────

with tab_unitaire:
    st.subheader("Vérifier une adresse")

    col1, col2, col3, col4 = st.columns([2, 4, 2, 3])
    with col1:
        u_numero = st.text_input("N°", placeholder="12 bis")
    with col2:
        u_rue = st.text_input("Voie", placeholder="rue des Lilas")
    with col3:
        u_cp = st.text_input("Code postal", placeholder="75019")
    with col4:
        u_ville = st.text_input("Ville", placeholder="Paris")

    go = st.button("🔎 Vérifier", type="primary", disabled=not qpv_loaded)
    if not qpv_loaded:
        st.warning("⬅ Charge d'abord le référentiel QPV via la barre latérale.")

    if go:
        addr = " ".join(x.strip() for x in [u_numero, u_rue, u_cp, u_ville] if x.strip())
        if not addr:
            st.error("Saisis au moins une partie de l'adresse.")
        else:
            with st.spinner("Géocodage et test QPV…"):
                geo_res = _geocode_cached(addr, u_cp.strip(), "")
                en_qpv, code_q, nom_q = find_qpv(
                    geo_res["lat"], geo_res["lon"], qpv_gdf, code_col, name_col, sindex
                )

            if geo_res["lat"] is None:
                st.error("❌ Adresse non trouvée par la BAN. Vérifie la saisie.")
                if geo_res["label"]:
                    st.caption(geo_res["label"])
            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    if en_qpv:
                        st.success("✅ **En QPV**")
                    else:
                        st.info("➖ **Hors QPV**")
                    st.metric("Score BAN", fmt_score(geo_res["score"]))
                with c2:
                    st.write(f"**Adresse normalisée (BAN)** : {geo_res['label']}")
                    st.write(f"**Coordonnées** : {geo_res['lat']:.6f}, {geo_res['lon']:.6f}")
                    if en_qpv:
                        st.write(f"**Code QPV** : `{code_q}`")
                        st.write(f"**Nom du quartier** : {nom_q}")

                # Carte pydeck avec polygones QPV autour du point
                geojson = _qpv_geojson_subset(qpv_gdf, geo_res["lat"], geo_res["lon"])
                layers = []
                if geojson is not None:
                    layers.append(pdk.Layer(
                        "GeoJsonLayer",
                        data=geojson,
                        stroked=True,
                        filled=True,
                        get_fill_color=[235, 96, 96, 70],
                        get_line_color=[170, 30, 30, 220],
                        line_width_min_pixels=2,
                        pickable=True,
                    ))
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame([{
                        "lat": geo_res["lat"],
                        "lon": geo_res["lon"],
                        "label": geo_res["label"],
                    }]),
                    get_position=["lon", "lat"],
                    get_fill_color=[34, 139, 230, 220],
                    get_radius=60,
                    radius_min_pixels=8,
                    radius_max_pixels=14,
                    pickable=True,
                ))
                st.pydeck_chart(pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(
                        latitude=geo_res["lat"], longitude=geo_res["lon"], zoom=14
                    ),
                    tooltip={"text": "{label}"},
                ))


# ── Onglet 2 : lot CSV ──────────────────────────────────────────────────────

EXAMPLE_CSV = """numero;rue;code_postal;ville
12;rue des Cités;93300;Aubervilliers
3;place de la Concorde;75008;Paris
45;avenue Jean Jaurès;93500;Pantin
1;rue de la République;69001;Lyon
8;cours Mirabeau;13100;Aix-en-Provence
22;rue du Faubourg Saint-Antoine;75011;Paris
"""

with tab_lot:
    st.subheader("Traiter un lot d'adresses")
    if not qpv_loaded:
        st.warning("⬅ Charge d'abord le référentiel QPV via la barre latérale.")

    col_up, col_ex = st.columns([4, 1])
    with col_up:
        csv_file = st.file_uploader(
            "Glisse ton CSV ici",
            type=["csv", "txt"],
            disabled=not qpv_loaded,
            help="Séparateur détecté automatiquement (; ou ,)",
        )
    with col_ex:
        st.write("")
        st.write("")
        if st.button("📋 Exemple", disabled=not qpv_loaded, help="Charger 6 adresses de test"):
            st.session_state["use_example"] = True
            st.session_state.pop("uploaded_bytes", None)

    raw: Optional[bytes] = None
    source_name = ""
    if csv_file is not None:
        raw = csv_file.read()
        source_name = csv_file.name
        st.session_state.pop("use_example", None)
    elif st.session_state.get("use_example"):
        raw = EXAMPLE_CSV.encode("utf-8")
        source_name = "exemple.csv"
        st.info("Mode exemple chargé. Clique sur **Lancer le traitement** ci-dessous.")

    if raw is not None and qpv_loaded:
        # Auto-détection encodage
        used_encoding = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                raw.decode(enc)
                used_encoding = enc
                break
            except UnicodeDecodeError:
                continue
        if used_encoding is None:
            used_encoding = "utf-8"

        sample = raw[:4096].decode(used_encoding, errors="replace")
        sep = ";" if sample.count(";") > sample.count(",") else ","
        st.caption(f"Source : `{source_name}` • Encodage : `{used_encoding}` • Séparateur : `{sep}`")

        try:
            df = pd.read_csv(
                io.BytesIO(raw), sep=sep, encoding=used_encoding, dtype=str
            ).fillna("")
        except Exception as e:
            st.error(f"Impossible de lire le CSV : {e}")
            st.stop()

        if df.empty:
            st.error("Le CSV est vide.")
            st.stop()

        st.write(f"**{len(df)}** ligne(s) lue(s). Aperçu :")
        st.dataframe(df.head(5), use_container_width=True)

        # ── Mapping des colonnes
        st.markdown("##### Mapping des colonnes")
        cols = ["(aucune)"] + list(df.columns)

        def guess(candidates):
            low = {c.lower(): c for c in df.columns}
            for cand in candidates:
                for k, v in low.items():
                    if cand in k:
                        return v
            return None

        guess_num   = guess(["numero", "num", "n°", "voie n", "numv"])
        guess_rue   = guess(["rue", "voie", "adresse", "libelle"])
        guess_cp    = guess(["cp", "code postal", "code_postal", "codepostal", "postal"])
        guess_ville = guess(["ville", "commune", "localite", "city"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            col_num = st.selectbox("N°", cols,
                                   index=cols.index(guess_num) if guess_num in cols else 0)
        with m2:
            col_rue = st.selectbox("Voie/rue", cols,
                                   index=cols.index(guess_rue) if guess_rue in cols else 0)
        with m3:
            col_cp = st.selectbox("Code postal", cols,
                                  index=cols.index(guess_cp) if guess_cp in cols else 0)
        with m4:
            col_ville = st.selectbox("Ville", cols,
                                     index=cols.index(guess_ville) if guess_ville in cols else 0)

        ready = col_rue != "(aucune)" and col_ville != "(aucune)"
        if not ready:
            st.info("Mappe au moins **voie** et **ville** pour pouvoir lancer.")

        use_batch = st.toggle(
            "⚡ Mode batch (recommandé)",
            value=True,
            help="Un seul appel à l'API BAN au lieu d'un par ligne. ~50× plus rapide.",
        )

        if st.button("🚀 Lancer le traitement", type="primary", disabled=not ready):
            t0 = time.time()
            session = requests.Session()

            address_cols = [c for c in [col_num, col_rue, col_ville] if c != "(aucune)"]
            postcode_col = col_cp if col_cp != "(aucune)" else ""

            # Adresse reconstituée pour traçabilité
            parts = []
            for c in [col_num, col_rue, col_cp, col_ville]:
                if c != "(aucune)":
                    parts.append(df[c].astype(str).str.strip())
            addr_series = parts[0]
            for p in parts[1:]:
                addr_series = addr_series + " " + p
            addr_series = addr_series.str.replace(r"\s+", " ", regex=True).str.strip()

            if use_batch:
                with st.spinner("Géocodage batch via l'API BAN…"):
                    try:
                        geo_df = geocode_batch_csv(
                            df,
                            address_cols=address_cols,
                            postcode_col=postcode_col,
                            citycode_col="",
                            session=session,
                        )
                    except Exception as e:
                        st.error(f"Échec du géocodage batch : {e}")
                        st.stop()

                with st.spinner("Test point-dans-polygone QPV…"):
                    qpv_df = find_qpv_batch(
                        geo_df["lat"], geo_df["lon"], qpv_gdf, code_col, name_col
                    )

                statut = []
                for la, oui in zip(geo_df["lat"], qpv_df["en_qpv"]):
                    if pd.isna(la):
                        statut.append("Adresse non géocodée")
                    elif bool(oui):
                        statut.append("Oui")
                    else:
                        statut.append("Non")

                enriched = df.copy()
                enriched["adresse_envoyee"] = addr_series
                enriched["adresse_ban"]     = geo_df["adresse_ban"]
                enriched["lat"]             = geo_df["lat"]
                enriched["lon"]             = geo_df["lon"]
                enriched["score_ban"]       = geo_df["score_ban"]
                enriched["en_qpv"]          = statut
                enriched["code_qpv"]        = qpv_df["code_qpv"]
                enriched["nom_qpv"]         = qpv_df["nom_qpv"]
            else:
                progress = st.progress(0.0, text="Initialisation…")
                results = []
                n = len(df)
                for i, (_, row) in enumerate(df.iterrows()):
                    cp_val = row[col_cp].strip() if col_cp != "(aucune)" else ""
                    addr = addr_series.iloc[i]
                    geo_res = geocode(addr, postcode=cp_val, citycode="", session=session)
                    en_qpv, code_q, nom_q = find_qpv(
                        geo_res["lat"], geo_res["lon"], qpv_gdf, code_col, name_col, sindex
                    )
                    if geo_res["lat"] is None:
                        st_v = "Adresse non géocodée"
                    elif en_qpv:
                        st_v = "Oui"
                    else:
                        st_v = "Non"
                    results.append({
                        "adresse_envoyee": addr,
                        "adresse_ban":     geo_res["label"],
                        "lat":             geo_res["lat"],
                        "lon":             geo_res["lon"],
                        "score_ban":       geo_res["score"],
                        "en_qpv":          st_v,
                        "code_qpv":        code_q,
                        "nom_qpv":         nom_q,
                    })
                    progress.progress((i + 1) / n, text=f"Ligne {i+1}/{n}")
                    time.sleep(0.05)

                enriched = pd.concat(
                    [df.reset_index(drop=True), pd.DataFrame(results)], axis=1
                )

            elapsed = time.time() - t0

            # ── Résumé
            nb_oui = int((enriched["en_qpv"] == "Oui").sum())
            nb_non = int((enriched["en_qpv"] == "Non").sum())
            nb_err = int((enriched["en_qpv"] == "Adresse non géocodée").sum())
            score_num = pd.to_numeric(enriched["score_ban"], errors="coerce")
            nb_low = int((score_num < 0.7).sum())

            st.success(f"Traitement terminé en **{elapsed:.1f} s** sur {len(enriched)} ligne(s).")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("En QPV", nb_oui)
            k2.metric("Hors QPV", nb_non)
            k3.metric("Non géocodées", nb_err)
            k4.metric("Score < 0.7", nb_low, help="À revérifier manuellement")

            st.markdown("##### Résultat")
            st.dataframe(
                enriched[[
                    "adresse_envoyee", "adresse_ban", "en_qpv",
                    "code_qpv", "nom_qpv", "score_ban",
                ]],
                use_container_width=True,
            )

            # ── Carte interactive
            geocoded = enriched[enriched["lat"].notna() & enriched["lon"].notna()].copy()
            if not geocoded.empty:
                st.markdown("##### Carte")
                def _color(row):
                    if row["en_qpv"] == "Oui":
                        return [76, 175, 80, 220]      # vert
                    if row["en_qpv"] == "Non":
                        return [33, 150, 243, 220]     # bleu
                    return [244, 67, 54, 220]          # rouge (non géocodée, ne devrait pas arriver ici)

                geocoded["color"] = geocoded.apply(_color, axis=1)
                geocoded["tooltip"] = (
                    geocoded["adresse_ban"].astype(str) + "\n"
                    + "QPV : " + geocoded["en_qpv"].astype(str)
                    + geocoded["nom_qpv"].fillna("").apply(
                        lambda x: f"\n{x}" if x else ""
                    )
                )

                qpv_geo = _qpv_geojson_for_points(qpv_gdf, geocoded)
                layers = []
                if qpv_geo is not None:
                    layers.append(pdk.Layer(
                        "GeoJsonLayer",
                        data=qpv_geo,
                        stroked=True,
                        filled=True,
                        get_fill_color=[235, 96, 96, 60],
                        get_line_color=[170, 30, 30, 200],
                        line_width_min_pixels=1,
                        pickable=False,
                    ))
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=geocoded[["lat", "lon", "color", "tooltip"]],
                    get_position=["lon", "lat"],
                    get_fill_color="color",
                    get_radius=80,
                    radius_min_pixels=5,
                    radius_max_pixels=12,
                    pickable=True,
                ))
                st.pydeck_chart(pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(
                        latitude=float(geocoded["lat"].mean()),
                        longitude=float(geocoded["lon"].mean()),
                        zoom=6 if len(geocoded) > 20 else 10,
                    ),
                    tooltip={"text": "{tooltip}"},
                ))

            # ── Téléchargement
            buf = io.StringIO()
            enriched.to_csv(buf, sep=sep, encoding="utf-8-sig", index=False)
            st.download_button(
                "⬇ Télécharger le CSV enrichi",
                data=buf.getvalue().encode("utf-8-sig"),
                file_name=f"{Path(source_name).stem}_qpv.csv",
                mime="text/csv",
                type="primary",
            )
