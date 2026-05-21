"""Tests unitaires de verif_qpv (sans réseau)."""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from verif_qpv import (
    FALLBACK_SCORE_THRESHOLD,
    QPV_CODE_CANDIDATES,
    QPV_NAME_CANDIDATES,
    _pick_shapefile_in_zip,
    apply_cadastre_fallback,
    build_address_series,
    detect_col,
    extract_parcelle_ref,
    find_qpv,
    find_qpv_batch,
    geocode,
    geocode_batch_csv,
    lookup_commune_insee,
    lookup_parcelle_coords,
    normalize_address,
    pick_shapefile_resource,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_qpv() -> gpd.GeoDataFrame:
    """2 polygones QPV factices, en WGS84 (lat/lon)."""
    p_paris = Polygon([(2.30, 48.85), (2.40, 48.85), (2.40, 48.90), (2.30, 48.90)])
    p_lyon = Polygon([(4.80, 45.74), (4.86, 45.74), (4.86, 45.78), (4.80, 45.78)])
    gdf = gpd.GeoDataFrame(
        {
            "code_qp": ["QP075001", "QP069001"],
            "noms_des_quartiers_prioritaires": ["Test Paris", "Test Lyon"],
            "geometry": [p_paris, p_lyon],
        },
        crs="EPSG:4326",
    )
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# detect_col
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectCol:
    def test_finds_canonical_name(self, fake_qpv):
        assert detect_col(fake_qpv, QPV_CODE_CANDIDATES) == "code_qp"
        assert detect_col(fake_qpv, QPV_NAME_CANDIDATES) == "noms_des_quartiers_prioritaires"

    def test_case_insensitive(self, fake_qpv):
        gdf = fake_qpv.rename(columns={"code_qp": "CODE_QP"})
        assert detect_col(gdf, QPV_CODE_CANDIDATES) == "CODE_QP"

    def test_returns_none_if_no_match(self, fake_qpv):
        gdf = fake_qpv.rename(columns={"code_qp": "totalement_inconnu"})
        assert detect_col(gdf, ["foo", "bar"]) is None

    def test_override_when_exists(self, fake_qpv):
        assert detect_col(fake_qpv, [], override="code_qp") == "code_qp"

    def test_override_raises_when_missing(self, fake_qpv):
        with pytest.raises(ValueError, match="n'existe pas"):
            detect_col(fake_qpv, [], override="colonne_qui_existe_pas")


# ─────────────────────────────────────────────────────────────────────────────
# find_qpv (unitaire)
# ─────────────────────────────────────────────────────────────────────────────

class TestFindQpv:
    def test_point_inside_polygon(self, fake_qpv):
        sindex = fake_qpv.sindex
        en_qpv, code, nom = find_qpv(
            48.87, 2.35, fake_qpv, "code_qp", "noms_des_quartiers_prioritaires", sindex
        )
        assert en_qpv is True
        assert code == "QP075001"
        assert nom == "Test Paris"

    def test_point_outside_polygon(self, fake_qpv):
        sindex = fake_qpv.sindex
        en_qpv, code, nom = find_qpv(
            48.80, 2.20, fake_qpv, "code_qp", "noms_des_quartiers_prioritaires", sindex
        )
        assert en_qpv is False
        assert code is None
        assert nom is None

    def test_handles_none_coords(self, fake_qpv):
        sindex = fake_qpv.sindex
        en_qpv, code, nom = find_qpv(
            None, None, fake_qpv, "code_qp", "noms_des_quartiers_prioritaires", sindex
        )
        assert en_qpv is False
        assert code is None and nom is None

    def test_handles_nan_coords(self, fake_qpv):
        sindex = fake_qpv.sindex
        en_qpv, _, _ = find_qpv(
            float("nan"), float("nan"), fake_qpv, "code_qp", None, sindex
        )
        assert en_qpv is False

    def test_works_without_name_col(self, fake_qpv):
        sindex = fake_qpv.sindex
        en_qpv, code, nom = find_qpv(48.87, 2.35, fake_qpv, "code_qp", None, sindex)
        assert en_qpv is True
        assert code == "QP075001"
        assert nom is None


# ─────────────────────────────────────────────────────────────────────────────
# find_qpv_batch (vectorisé)
# ─────────────────────────────────────────────────────────────────────────────

class TestFindQpvBatch:
    def test_mixed_points(self, fake_qpv):
        lat = pd.Series([48.87, 45.76, 48.80, None])
        lon = pd.Series([2.35, 4.83, 2.20, None])
        out = find_qpv_batch(lat, lon, fake_qpv, "code_qp", "noms_des_quartiers_prioritaires")

        assert list(out["en_qpv"]) == [True, True, False, False]
        assert out.loc[0, "code_qpv"] == "QP075001"
        assert out.loc[1, "code_qpv"] == "QP069001"
        assert out.loc[2, "code_qpv"] is None or pd.isna(out.loc[2, "code_qpv"])

    def test_all_none(self, fake_qpv):
        lat = pd.Series([None, None])
        lon = pd.Series([None, None])
        out = find_qpv_batch(lat, lon, fake_qpv, "code_qp", "noms_des_quartiers_prioritaires")
        assert list(out["en_qpv"]) == [False, False]

    def test_preserves_index(self, fake_qpv):
        lat = pd.Series([48.87, 45.76], index=[100, 200])
        lon = pd.Series([2.35, 4.83], index=[100, 200])
        out = find_qpv_batch(lat, lon, fake_qpv, "code_qp", None)
        assert list(out.index) == [100, 200]


# ─────────────────────────────────────────────────────────────────────────────
# geocode (mocké)
# ─────────────────────────────────────────────────────────────────────────────

class TestGeocode:
    def _make_session(self, json_response: dict, status: int = 200):
        sess = MagicMock()
        resp = MagicMock()
        resp.status_code = status
        resp.raise_for_status = MagicMock()
        resp.json.return_value = json_response
        sess.get.return_value = resp
        return sess

    def test_returns_lat_lon_score(self):
        session = self._make_session({
            "features": [{
                "geometry": {"coordinates": [2.349, 48.864]},
                "properties": {"score": 0.92, "label": "1 rue X 75001 Paris"},
            }]
        })
        out = geocode("1 rue X 75001 Paris", session=session)
        assert out["lat"] == 48.864
        assert out["lon"] == 2.349
        assert out["score"] == 0.92
        assert out["label"] == "1 rue X 75001 Paris"

    def test_empty_query_returns_empty(self):
        session = MagicMock()
        out = geocode("", session=session)
        assert out["lat"] is None
        assert out["lon"] is None
        session.get.assert_not_called()

    def test_no_features_returns_empty(self):
        session = self._make_session({"features": []})
        out = geocode("xyz inexistant", session=session)
        assert out["lat"] is None
        assert out["lon"] is None

    def test_api_error_returns_error_label(self):
        sess = MagicMock()
        sess.get.side_effect = RuntimeError("boom")
        out = geocode("1 rue X", session=sess)
        assert out["lat"] is None
        assert "ERREUR" in (out["label"] or "")

    def test_passes_postcode_param(self):
        session = self._make_session({"features": []})
        geocode("rue X", postcode="75001", session=session)
        call_kwargs = session.get.call_args.kwargs
        assert call_kwargs["params"]["postcode"] == "75001"


# ─────────────────────────────────────────────────────────────────────────────
# geocode_batch_csv (mocké)
# ─────────────────────────────────────────────────────────────────────────────

class TestGeocodeBatchCsv:
    def test_parses_response(self):
        df = pd.DataFrame({
            "numero": ["1", "2"],
            "rue": ["rue A", "rue B"],
            "cp": ["75001", "75002"],
            "ville": ["Paris", "Paris"],
        })
        # La réponse BAN reprend les colonnes envoyées : _row_id, address, postcode_v + result_*
        response_csv = (
            "_row_id,address,postcode_v,result_label,result_score,latitude,longitude\n"
            "0,1 rue A 75001 Paris,75001,1 rue A 75001 Paris,0.91,48.860,2.340\n"
            "1,2 rue B 75002 Paris,75002,2 rue B 75002 Paris,0.88,48.870,2.350\n"
        )
        sess = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.content = response_csv.encode("utf-8")
        sess.post.return_value = resp

        out = geocode_batch_csv(
            df,
            address_cols=["numero", "rue", "ville"],
            postcode_col="cp",
            session=sess,
        )
        assert len(out) == 2
        assert out.iloc[0]["lat"] == 48.860
        assert out.iloc[0]["lon"] == 2.340
        assert out.iloc[0]["score_ban"] == 0.91
        assert out.iloc[0]["adresse_ban"] == "1 rue A 75001 Paris"
        assert out.iloc[0]["adresse_envoyee"] == "1 rue A Paris"
        # On envoie maintenant une seule colonne 'address' à la BAN
        call = sess.post.call_args
        sent_data = call.kwargs.get("data") or []
        assert ("columns", "address") in sent_data
        assert ("postcode", "postcode_v") in sent_data

    def test_normalizes_double_numbers_in_payload(self):
        """Vérifie que '8/10 rue X' devient '8 rue X' dans le CSV envoyé."""
        df = pd.DataFrame({"adresse": ["8/10 rue des Champs Paris"]})
        response_csv = (
            "_row_id,address,result_label,result_score,latitude,longitude\n"
            "0,8 rue des Champs Paris,8 rue des Champs 75008 Paris,0.95,48.87,2.32\n"
        )
        sess = MagicMock()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.content = response_csv.encode("utf-8")
        sess.post.return_value = resp

        out = geocode_batch_csv(df, address_cols=["adresse"], session=sess)
        # Le CSV envoyé en multipart contient l'adresse normalisée
        files = sess.post.call_args.kwargs["files"]
        sent_csv = files["data"][1].decode("utf-8")
        assert "8 rue des Champs Paris" in sent_csv
        assert "8/10" not in sent_csv
        # adresse_envoyee dans le résultat reflète aussi la normalisation
        assert out.iloc[0]["adresse_envoyee"] == "8 rue des Champs Paris"

    def test_empty_df_returns_empty(self):
        sess = MagicMock()
        out = geocode_batch_csv(pd.DataFrame(columns=["a"]), ["a"], session=sess)
        assert len(out) == 0
        sess.post.assert_not_called()

    def test_raises_if_no_address_cols(self):
        with pytest.raises(ValueError):
            geocode_batch_csv(pd.DataFrame({"a": [1]}), address_cols=[])


# ─────────────────────────────────────────────────────────────────────────────
# pick_shapefile_resource
# ─────────────────────────────────────────────────────────────────────────────

class TestPickShapefileResource:
    def test_picks_zipped_shapefile(self):
        resources = [
            {"title": "Données tabulaires (CSV)", "format": "csv",
             "url": "https://x/data.csv", "filesize": 1000},
            {"title": "Shapefile France hexagonale 2024", "format": "shp",
             "url": "https://x/qpv.zip", "filesize": 50_000_000},
            {"title": "GeoJSON", "format": "geojson",
             "url": "https://x/qpv.geojson", "filesize": 10_000_000},
        ]
        res = pick_shapefile_resource(resources)
        assert res is not None
        assert res["url"] == "https://x/qpv.zip"

    def test_returns_none_if_no_match(self):
        resources = [
            {"title": "CSV", "format": "csv", "url": "https://x/a.csv", "filesize": 1},
        ]
        assert pick_shapefile_resource(resources) is None

    def test_prefers_shp_only_over_combined_pack(self):
        """Le pack 'SHP seul' doit gagner contre 'GEOJSON-GPKG-SHP combiné'."""
        resources = [
            {"title": "Périmètres des quartiers prioritaires 2024 (GEOJSON - GPKG - SHP)",
             "format": "zip", "url": "https://x/qpv-2024.zip", "filesize": 16_800_000},
            {"title": "Périmètre des QP 2024 (SHP)",
             "format": "zip", "url": "https://x/qpv-2024-shp.zip", "filesize": 5_800_000},
            {"title": "Périmètre des QP 2024 (format GPKG)",
             "format": "zip", "url": "https://x/qpv-2024-gpkg.zip", "filesize": 6_000_000},
        ]
        res = pick_shapefile_resource(resources)
        assert res is not None
        assert res["url"] == "https://x/qpv-2024-shp.zip"

    def test_prefers_2024_over_2015(self):
        resources = [
            {"title": "Périmètre des QP 2015 (format shapefile)",
             "format": "zip", "url": "https://x/qp-2015-shp.zip", "filesize": 3_800_000},
            {"title": "Périmètre des QP 2024 (SHP)",
             "format": "zip", "url": "https://x/qpv-2024-shp.zip", "filesize": 5_800_000},
        ]
        res = pick_shapefile_resource(resources)
        assert res["url"] == "https://x/qpv-2024-shp.zip"


# ─────────────────────────────────────────────────────────────────────────────
# _pick_shapefile_in_zip
# ─────────────────────────────────────────────────────────────────────────────

class TestPickShapefileInZip:
    def test_picks_hexagone_outre_mer_wgs84(self):
        """Reproduit la structure réelle du zip de l'ANCT."""
        names = [
            "SHP/",
            "SHP/QP2024_France_hexagonale_LB93.shp",
            "SHP/QP2024_France_Hexagonale_Outre_Mer_WGS84.shp",
            "SHP/QP2024_Guadeloupe_RGAF09_UTM20N.shp",
            "SHP/QP2024_Guyane_RGF95_UTM22N.shp",
            "SHP/QP2024_LaReunion_RGR92_UTM40S.shp",
        ]
        assert _pick_shapefile_in_zip(names) == "SHP/QP2024_France_Hexagonale_Outre_Mer_WGS84.shp"

    def test_single_shapefile_returns_it(self):
        assert _pick_shapefile_in_zip(["data.shp", "data.dbf"]) == "data.shp"

    def test_no_shapefile_returns_none(self):
        assert _pick_shapefile_in_zip(["data.geojson", "readme.txt"]) is None


# ─────────────────────────────────────────────────────────────────────────────
# normalize_address
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeAddress:
    @pytest.mark.parametrize("addr, expected", [
        ("8/10 rue des Champs",       "8 rue des Champs"),
        ("8-10 rue des Champs",       "8 rue des Champs"),
        ("8 / 10 rue des Champs",     "8 rue des Champs"),
        ("8 - 10 rue X",              "8 rue X"),
        ("8 et 10 rue X",             "8 rue X"),
        ("8 à 10 rue X",              "8 rue X"),
        ("8 a 10 rue X",              "8 rue X"),
        ("8 bis/10 rue X",            "8 bis rue X"),
        ("8 ter - 10 rue X",          "8 ter rue X"),
        ("8 bis et 10 ter rue X",     "8 bis rue X"),
        ("  8/10 rue X",              "8 rue X"),
    ])
    def test_double_number(self, addr, expected):
        assert normalize_address(addr) == expected

    @pytest.mark.parametrize("addr", [
        "12 rue saint-michel",          # tiret dans le nom de rue
        "rue de la Paix",                # pas de numéro
        "1 rue Notre-Dame-de-Lorette",   # tirets dans le nom
        "5 place de la République 75011 Paris",
        "8 bis rue X",                   # numéro+suffixe seul
    ])
    def test_unchanged(self, addr):
        assert normalize_address(addr) == addr

    def test_empty(self):
        assert normalize_address("") == ""

    def test_collapses_whitespace(self):
        assert normalize_address("8/10   rue   des   Champs") == "8 rue des Champs"

    # ── Cas Emmy : annotations cadastrales + adresse dupliquée ─────────────
    def test_emmy_strips_parcelle_and_dedupes(self):
        addr = (
            "42 BOULEVARD JEROME TRESSAGUET - Parcelle : 000 , CR ,0694 "
            "42 BOULEVARD JEROME TRESSAGUET - Parcelle : 000 , CR ,0694 "
            "58000 Nevers"
        )
        assert normalize_address(addr) == "42 BOULEVARD JEROME TRESSAGUET 58000 Nevers"

    def test_emmy_parcelle_with_slash_refs(self):
        addr = (
            "39 AVENUE JEAN MOULIN - Parcelle : 000 / AB / 0119 "
            "39 AVENUE JEAN MOULIN - Parcelle : 000 / AB / 0119 "
            "24700 Montpon-Menesterol"
        )
        assert normalize_address(addr) == "39 AVENUE JEAN MOULIN 24700 Montpon-Menesterol"

    def test_emmy_parcelle_with_multinum(self):
        addr = (
            "7 rue de corse - Parcelle : 000/DX/0072 "
            "7/9 rue de corse - Parcelle : 000/DX/0072 "
            "93600 AULNAY sous bois"
        )
        assert normalize_address(addr) == "7 rue de corse 93600 AULNAY sous bois"

    def test_emmy_dedupe_with_multinum_in_middle(self):
        addr = "107 RUE GALLIEN 107/113 RUE GALLIEN 93000 BOBIGNY"
        assert normalize_address(addr) == "107 RUE GALLIEN 93000 BOBIGNY"

    def test_multinum_anywhere_not_just_start(self):
        """Le double numéro est normalisé même au milieu de l'adresse."""
        assert normalize_address("résidence A 8/10 rue des Lilas") == "résidence A 8 rue des Lilas"


# ─────────────────────────────────────────────────────────────────────────────
# build_address_series
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAddressSeries:
    def test_combines_columns_and_normalizes(self):
        df = pd.DataFrame({
            "numero": ["8/10", "12", ""],
            "rue":    ["rue X", "rue Y", "rue Z"],
            "cp":     ["75001", "75002", "75003"],
            "ville":  ["Paris", "Paris", "Paris"],
        })
        out = build_address_series(df, ["numero", "rue", "cp", "ville"])
        assert out.tolist() == [
            "8 rue X 75001 Paris",
            "12 rue Y 75002 Paris",
            "rue Z 75003 Paris",
        ]

    def test_can_skip_normalization(self):
        df = pd.DataFrame({"a": ["8/10 rue X"]})
        assert build_address_series(df, ["a"], normalize=False).iloc[0] == "8/10 rue X"

    def test_handles_nan(self):
        df = pd.DataFrame({"a": [None, "12"], "b": ["rue X", "rue Y"]})
        out = build_address_series(df, ["a", "b"])
        assert out.tolist() == ["rue X", "12 rue Y"]

    def test_raises_on_empty_cols(self):
        with pytest.raises(ValueError):
            build_address_series(pd.DataFrame({"a": [1]}), [])


# ─────────────────────────────────────────────────────────────────────────────
# extract_parcelle_ref
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractParcelleRef:
    @pytest.mark.parametrize("addr, expected", [
        (
            "7 rue de corse - Parcelle : 000/DX/0072 93600 AULNAY",
            {"prefixe": "000", "section": "DX", "numero": "0072"},
        ),
        (
            "Parcelle : 000 , CR ,0694 58000 Nevers",
            {"prefixe": "000", "section": "CR", "numero": "0694"},
        ),
        (
            "39 AVENUE JEAN MOULIN - Parcelle : 000 / AB / 0119 24700",
            {"prefixe": "000", "section": "AB", "numero": "0119"},
        ),
    ])
    def test_extracts(self, addr, expected):
        assert extract_parcelle_ref(addr) == expected

    @pytest.mark.parametrize("addr", [
        "",
        "12 rue saint-michel 75001 Paris",   # pas de parcelle
        "Parcelle : trop court",              # pas de format reconnu
    ])
    def test_none_when_no_ref(self, addr):
        assert extract_parcelle_ref(addr) is None

    def test_pads_numero_to_4_digits(self):
        result = extract_parcelle_ref("Parcelle : 000/AB/9")
        assert result["numero"] == "0009"

    def test_uppercases_section(self):
        result = extract_parcelle_ref("Parcelle : 000/ab/0119")
        assert result["section"] == "AB"


# ─────────────────────────────────────────────────────────────────────────────
# lookup_commune_insee (BAN mockée)
# ─────────────────────────────────────────────────────────────────────────────

class TestLookupCommuneInsee:
    def _mock_session(self, citycode):
        sess = MagicMock()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "features": [{"properties": {"citycode": citycode}}] if citycode else []
        }
        sess.get.return_value = resp
        return sess

    def test_returns_insee_code(self):
        sess = self._mock_session("75056")
        assert lookup_commune_insee("75001", "Paris", session=sess) == "75056"

    def test_returns_none_if_no_match(self):
        sess = self._mock_session(None)
        assert lookup_commune_insee("99999", "Inconnu", session=sess) is None

    def test_returns_none_on_empty_input(self):
        sess = MagicMock()
        assert lookup_commune_insee("", "", session=sess) is None
        sess.get.assert_not_called()

    def test_passes_type_municipality(self):
        sess = self._mock_session("75056")
        lookup_commune_insee("75001", "Paris", session=sess)
        params = sess.get.call_args.kwargs["params"]
        assert params["type"] == "municipality"


# ─────────────────────────────────────────────────────────────────────────────
# lookup_parcelle_coords (API Carto mockée)
# ─────────────────────────────────────────────────────────────────────────────

class TestLookupParcelleCoords:
    def _mock_session(self, geojson_features):
        sess = MagicMock()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"features": geojson_features}
        sess.get.return_value = resp
        return sess

    def test_returns_centroid(self):
        # Polygone carré 2.30,48.85 – 2.40,48.90 → centroïde 2.35, 48.875
        features = [{
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [2.30, 48.85], [2.40, 48.85],
                    [2.40, 48.90], [2.30, 48.90], [2.30, 48.85],
                ]],
            }
        }]
        sess = self._mock_session(features)
        coords = lookup_parcelle_coords("75056", "AB", "0119", session=sess)
        assert coords is not None
        assert abs(coords["lat"] - 48.875) < 1e-6
        assert abs(coords["lon"] - 2.35) < 1e-6

    def test_returns_none_when_no_feature(self):
        sess = self._mock_session([])
        assert lookup_parcelle_coords("75056", "AB", "0119", session=sess) is None

    def test_pads_section_to_2_chars(self):
        sess = self._mock_session([])
        lookup_parcelle_coords("75056", "A", "0119", session=sess)
        params = sess.get.call_args.kwargs["params"]
        assert params["section"] == "0A"

    def test_returns_none_on_missing_args(self):
        sess = MagicMock()
        assert lookup_parcelle_coords("", "AB", "0119", session=sess) is None
        sess.get.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# apply_cadastre_fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyCadastreFallback:
    def test_replaces_low_score_rows_with_cadastre_coords(self, monkeypatch):
        geo_df = pd.DataFrame({
            "adresse_envoyee": ["rue X 75001 Paris", "lieu-dit 24700"],
            "adresse_ban":     ["1 rue X 75001 Paris", "?"],
            "lat":             [48.86, 45.10],
            "lon":             [2.35, 0.50],
            "score_ban":       [0.92, 0.40],     # ligne 1 OK, ligne 2 < seuil
        })
        parcelles = pd.Series([
            None,
            {"prefixe": "000", "section": "AB", "numero": "0119"},
        ])
        postcodes = pd.Series(["75001", "24700"])
        cities = pd.Series(["Paris", "Montpon-Ménestérol"])

        # Mock les lookups
        from verif_qpv import lookup_commune_insee, lookup_parcelle_coords
        called_with = {}

        def fake_insee(cp, ville, session=None):
            called_with["insee"] = (cp, ville)
            return "24279"

        def fake_parcelle(insee, section, numero, session=None):
            called_with["parcelle"] = (insee, section, numero)
            return {"lat": 45.20, "lon": 0.55}

        monkeypatch.setattr("verif_qpv.lookup_commune_insee", fake_insee)
        monkeypatch.setattr("verif_qpv.lookup_parcelle_coords", fake_parcelle)
        monkeypatch.setattr("verif_qpv.time.sleep", lambda x: None)

        out = apply_cadastre_fallback(
            geo_df, parcelles, postcodes, cities,
            score_threshold=FALLBACK_SCORE_THRESHOLD,
            session=MagicMock(),
        )

        # Ligne 0 inchangée (BAN bon)
        assert out.loc[0, "source_geocodage"] == "BAN"
        assert out.loc[0, "lat"] == 48.86
        # Ligne 1 remplacée par cadastre
        assert out.loc[1, "source_geocodage"] == "Cadastre"
        assert out.loc[1, "lat"] == 45.20
        assert out.loc[1, "lon"] == 0.55
        assert called_with["parcelle"] == ("24279", "AB", "0119")

    def test_no_parcelle_keeps_ban_result(self, monkeypatch):
        geo_df = pd.DataFrame({
            "adresse_envoyee": ["X"],
            "adresse_ban":     ["X"],
            "lat":             [48.86],
            "lon":             [2.35],
            "score_ban":       [0.30],
        })
        parcelles = pd.Series([None])  # pas de parcelle
        postcodes = pd.Series([""])
        cities = pd.Series([""])
        monkeypatch.setattr("verif_qpv.time.sleep", lambda x: None)
        out = apply_cadastre_fallback(geo_df, parcelles, postcodes, cities, session=MagicMock())
        # Aucune tentative de cadastre, mais on garde "BAN" comme source
        assert out.loc[0, "source_geocodage"] == "BAN"
        assert out.loc[0, "lat"] == 48.86

    def test_marks_failed_fallback(self, monkeypatch):
        geo_df = pd.DataFrame({
            "adresse_envoyee": ["X"],
            "adresse_ban":     ["X"],
            "lat":             [48.86],
            "lon":             [2.35],
            "score_ban":       [0.30],
        })
        parcelles = pd.Series([{"prefixe": "000", "section": "AB", "numero": "0119"}])
        postcodes = pd.Series(["75001"])
        cities = pd.Series(["Paris"])

        monkeypatch.setattr("verif_qpv.lookup_commune_insee", lambda *a, **k: "75056")
        monkeypatch.setattr("verif_qpv.lookup_parcelle_coords", lambda *a, **k: None)
        monkeypatch.setattr("verif_qpv.time.sleep", lambda x: None)

        out = apply_cadastre_fallback(geo_df, parcelles, postcodes, cities, session=MagicMock())
        assert out.loc[0, "source_geocodage"] == "BAN (cadastre indisponible)"
        assert out.loc[0, "lat"] == 48.86  # inchangé
