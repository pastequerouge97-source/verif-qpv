#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verif_qpv.py — Vérification automatique de l'appartenance à un QPV
==================================================================

Pour chaque ligne d'un CSV (typiquement un export Emmy), le script :
  1. Reconstitue l'adresse à partir de plusieurs colonnes (num, rue, CP, ville).
  2. Géocode l'adresse via l'API publique BAN (api-adresse.data.gouv.fr).
     - Mode unitaire (un appel par ligne) ou batch (un seul appel pour tout le CSV).
  3. Vérifie si le point obtenu tombe dans un polygone QPV 2024
     (référentiel à télécharger une fois depuis data.gouv.fr / sig.ville.gouv.fr).
  4. Écrit un CSV enrichi avec les colonnes :
        - adresse_envoyee     : adresse reconstituée
        - adresse_ban         : libellé renvoyé par la BAN
        - lat, lon            : coordonnées WGS84
        - score_ban           : confiance du géocodage (0 à 1, > 0.7 = fiable)
        - en_qpv              : "Oui" / "Non" / "Adresse non géocodée"
        - code_qpv            : code officiel du QPV (ex. QN075033)
        - nom_qpv             : nom du quartier prioritaire

Dépendances :
    pip install pandas geopandas requests

Exemples :
    # Mode batch (recommandé, beaucoup plus rapide) :
    python verif_qpv.py --input dossiers.csv --output enrichi.csv \\
        --qpv QP2024_France_hexagonale.shp --batch

    # Mode unitaire (legacy, plus lent mais détaillé) :
    python verif_qpv.py --input dossiers.csv --output enrichi.csv \\
        --qpv QP2024_France_hexagonale.shp \\
        --col-numero "N°" --col-rue "Voie" --col-cp "CP" --col-ville "Commune"

Source officielle des polygones QPV 2024 :
    https://www.data.gouv.fr/datasets/quartiers-prioritaires-de-la-politique-de-la-ville-qpv
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point, shape


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BAN_URL = "https://api-adresse.data.gouv.fr/search/"
BAN_CSV_URL = "https://api-adresse.data.gouv.fr/search/csv/"
APICARTO_PARCELLE_URL = "https://apicarto.ign.fr/api/cadastre/parcelle"
REQUEST_TIMEOUT = 10           # secondes (mode unitaire)
BATCH_TIMEOUT = 180            # secondes (mode batch, plus long sur gros fichier)
BATCH_MAX_ROWS = 10_000        # on découpe au-delà pour rester sous les limites
SLEEP_BETWEEN_REQUESTS = 0.05  # l'API BAN tolère ~50 req/s ; on reste prudent
SLEEP_BETWEEN_CADASTRE = 0.2   # API Carto plus stricte, on espace
FALLBACK_SCORE_THRESHOLD = 0.70  # en dessous, on tente le fallback cadastre si parcelle dispo

# Référentiel QPV — data.gouv.fr
# (l'API v1 renvoie les resources en liste, contrairement à v2 où c'est une sous-ressource)
QPV_DATASET_SLUG = "quartiers-prioritaires-de-la-politique-de-la-ville-qpv"
QPV_DATASET_API = f"https://www.data.gouv.fr/api/1/datasets/{QPV_DATASET_SLUG}/"

QPV_CODE_CANDIDATES = [
    "code_qp", "Code_QP", "CODE_QP", "id_qp", "qp_code", "CODE", "code",
]
QPV_NAME_CANDIDATES = [
    "noms_des_quartiers_prioritaires", "nom_qp", "NOM_QP",
    "lib_qp", "libelle", "LIB_QP", "nom", "NOM",
]


# ---------------------------------------------------------------------------
# Normalisation d'adresse
# ---------------------------------------------------------------------------

# Mots-clés de type de voie en France (utilisés pour détecter le début d'une adresse).
_STREET_TYPES = (
    r"rue|av|avenue|bd|boulevard|impasse|imp|place|pl|all[ée]e|chemin|ch|route|"
    r"cours|quai|villa|voie|passage|sentier|cit[eé]|square|esplanade|faubourg|parc|"
    r"r[ée]sidence|lotissement|hameau|mont[ée]e|mte|traverse|carrefour"
)

# Marqueurs de fin d'annotation cadastrale : un postcode 5 chiffres,
# un pattern "numéro(s) + type de voie", ou la fin de la chaîne.
# Les sous-numéros utilisent / ou - mais PAS d'espace (sinon on capture trop loin).
_END_OF_METADATA = (
    r"(?="
    r"\s+\d{5}\b"
    rf"|\s+\d+(?:[/\-]\d+)*(?:\s+(?:bis|ter|quater))?\s+(?:{_STREET_TYPES})\b"
    r"|$"
    r")"
)

# Annotations Emmy / cadastre — "Parcelle : 000/AB/0119", "Références cadastrales : ...",
# "CR ,0694", etc.
_METADATA_RE = re.compile(
    r"\s*[-,;]?\s*"
    r"(?:parcelles?|r[ée]f[ée]rences?\s+cadastrales?|cadastr[ée]e?s?)"
    r"\s*:?\s*"
    r"[\w/.,\s\-]+?"
    + _END_OF_METADATA,
    flags=re.IGNORECASE,
)

# Double numéro : "8/10 rue X", "8-10 rue X", "8 et 10 rue X", "8 à 10 rue X",
# "8 bis/10 rue X"... — partout dans l'adresse, pas seulement en tête.
_MULTI_NUM_RE = re.compile(
    r"(?<![\d/])"                                 # pas précédé d'un chiffre ou d'un /
    r"(\d+(?:\s+(?:bis|ter|quater))?)"            # group 1 : premier numéro (+suffixe)
    r"\s*(?:[/\-]|\s+(?:et|à|a)\s+)\s*"           # séparateur : / - ' et ' ' à ' ' a '
    r"\d+(?:\s+(?:bis|ter|quater))?"              # second numéro (à supprimer)
    r"(?=\s+\D|$)",                                # suivi d'un espace+non-chiffre ou fin
    flags=re.IGNORECASE,
)


# Référence cadastrale dans l'adresse Emmy : "Parcelle : 000/AB/0119",
# "Parcelle : 000 , CR ,0694", "Parcelle : 000 / DX / 0072", etc.
# Format attendu : prefix (1-3 chiffres) / section (1-3 alphanum) / numéro (1-4 chiffres)
_PARCELLE_REF_RE = re.compile(
    r"parcelles?\s*:?\s*"
    r"(\d{1,3})\s*[/,\-\s]+\s*"
    r"([A-Z0-9]{1,3})\s*[/,\-\s]+\s*"
    r"(\d{1,4})\b",
    flags=re.IGNORECASE,
)


def extract_parcelle_ref(addr: str) -> Optional[dict]:
    """Extrait la référence cadastrale (préfixe/section/numéro) d'une adresse Emmy.

    Reconnaît les formats : "Parcelle : 000/AB/0119", "Parcelle : 000 , CR ,0694",
    "Parcelle : 000 / DX / 0072".

    Renvoie {"prefixe": "000", "section": "AB", "numero": "0119"} ou None
    si aucune référence n'est trouvée. Le numéro est complété à 4 chiffres
    (API Carto attend ce format).
    """
    if not addr:
        return None
    m = _PARCELLE_REF_RE.search(addr)
    if not m:
        return None
    prefixe = m.group(1).zfill(3)
    section = m.group(2).upper()
    numero = m.group(3).zfill(4)
    return {"prefixe": prefixe, "section": section, "numero": numero}


def _strip_emmy_metadata(addr: str) -> str:
    """Supprime les annotations Emmy/cadastre (Parcelle, références cadastrales)."""
    return _METADATA_RE.sub(" ", addr)


def _dedupe_consecutive(addr: str) -> str:
    """Supprime une répétition consécutive de ≥3 tokens identiques.

    Itère jusqu'à stabilisation pour gérer les triples (A B C A B C A B C → A B C).
        "42 BOULEVARD JEROME 42 BOULEVARD JEROME 58000 Nevers"
            → "42 BOULEVARD JEROME 58000 Nevers"
    """
    prev = None
    while addr != prev:
        prev = addr
        tokens = addr.split()
        n = len(tokens)
        found = False
        for w in range(min(n // 2, 10), 2, -1):
            for i in range(n - 2 * w + 1):
                if tokens[i : i + w] == tokens[i + w : i + 2 * w]:
                    addr = " ".join(tokens[: i + w] + tokens[i + 2 * w :])
                    found = True
                    break
            if found:
                break
    return addr


def normalize_address(addr: str) -> str:
    """Normalise une adresse pour le géocodage BAN.

    Étapes :
    1. Supprime les annotations Emmy/cadastre ("Parcelle : 000/AB/0119", etc.).
    2. Réduit les doubles numéros : "8/10", "8-10", "8 et 10", "8 à 10" → "8".
       Le suffixe bis/ter/quater du premier numéro est préservé.
    3. Supprime les répétitions consécutives de portions d'adresse
       (cas Emmy : la même adresse dupliquée dans deux colonnes mappées).
    4. Normalise les espaces multiples.

    Exemples :
        "42 BOULEVARD JEROME - Parcelle : 000, CR ,0694 42 BOULEVARD JEROME 58000 Nevers"
            → "42 BOULEVARD JEROME 58000 Nevers"
        "8/10 rue des Champs 75008 Paris"
            → "8 rue des Champs 75008 Paris"
        "12 rue saint-michel"
            → "12 rue saint-michel"  (inchangé)
    """
    if not addr:
        return addr
    out = _strip_emmy_metadata(addr)
    out = _MULTI_NUM_RE.sub(r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = _dedupe_consecutive(out)
    return out


def build_address_series(
    df: pd.DataFrame,
    address_cols: list[str],
    normalize: bool = True,
) -> pd.Series:
    """Concatène plusieurs colonnes en une adresse unique, optionnellement normalisée."""
    if not address_cols:
        raise ValueError("address_cols ne peut pas être vide")
    parts = [df[c].fillna("").astype(str).str.strip() for c in address_cols]
    addr = parts[0]
    for p in parts[1:]:
        addr = addr + " " + p
    addr = addr.str.replace(r"\s+", " ", regex=True).str.strip()
    if normalize:
        addr = addr.map(normalize_address)
    return addr


# ---------------------------------------------------------------------------
# Géocodage — mode unitaire
# ---------------------------------------------------------------------------

def geocode(
    query: str,
    postcode: str = "",
    citycode: str = "",
    session: Optional[requests.Session] = None,
) -> dict:
    """Appelle l'API BAN sur une adresse unitaire et renvoie un dict normalisé.

    L'adresse est normalisée avant envoi (gestion des doubles numéros "8/10 rue X").
    """
    empty = {"lat": None, "lon": None, "score": None, "label": None}
    if not query or not query.strip():
        return empty
    query = normalize_address(query)
    if session is None:
        session = requests.Session()
    params = {"q": query, "limit": 1, "autocomplete": 0}
    if postcode:
        params["postcode"] = postcode
    if citycode:
        params["citycode"] = citycode
    try:
        r = session.get(BAN_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {**empty, "label": f"ERREUR API : {e}"}
    feats = data.get("features") or []
    if not feats:
        return empty
    feat = feats[0]
    lon, lat = feat["geometry"]["coordinates"]
    props = feat.get("properties", {})
    return {
        "lat": lat,
        "lon": lon,
        "score": props.get("score"),
        "label": props.get("label"),
    }


# ---------------------------------------------------------------------------
# Géocodage — mode batch CSV (beaucoup plus rapide)
# ---------------------------------------------------------------------------

def geocode_batch_csv(
    df: pd.DataFrame,
    address_cols: list[str],
    postcode_col: str = "",
    citycode_col: str = "",
    session: Optional[requests.Session] = None,
    progress_cb=None,
) -> pd.DataFrame:
    """Géocode un DataFrame entier via l'endpoint /search/csv/ de la BAN.

    Les adresses sont concaténées depuis `address_cols`, normalisées
    (doubles numéros "8/10" → "8"), puis envoyées à la BAN en un seul appel.

    Renvoie un DataFrame indexé comme `df` avec les colonnes :
        adresse_envoyee (str), adresse_ban (str), lat (float), lon (float), score_ban (float)

    Si `df` dépasse BATCH_MAX_ROWS, il est découpé en plusieurs requêtes.
    `progress_cb(done, total)` est appelé à chaque chunk si fourni.
    """
    if not address_cols:
        raise ValueError("address_cols ne peut pas être vide")
    if session is None:
        session = requests.Session()

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["adresse_envoyee", "adresse_ban", "lat", "lon", "score_ban"])

    chunks = []
    total_chunks = (n + BATCH_MAX_ROWS - 1) // BATCH_MAX_ROWS
    for i in range(0, n, BATCH_MAX_ROWS):
        sub = df.iloc[i : i + BATCH_MAX_ROWS]
        chunk_out = _geocode_one_chunk(
            sub, address_cols, postcode_col, citycode_col, session
        )
        chunks.append(chunk_out)
        if progress_cb is not None:
            progress_cb(min(i + BATCH_MAX_ROWS, n), n)

    out = pd.concat(chunks, axis=0)
    out.index = df.index
    return out


def _geocode_one_chunk(
    df: pd.DataFrame,
    address_cols: list[str],
    postcode_col: str,
    citycode_col: str,
    session: requests.Session,
) -> pd.DataFrame:
    """Envoie un chunk au endpoint batch BAN et parse la réponse.

    Construit une colonne 'address' normalisée (gestion des doubles numéros)
    avant l'envoi, plutôt que de laisser la BAN concaténer plusieurs colonnes.
    """
    addr_series = build_address_series(df, address_cols, normalize=True)

    payload_dict: dict = {
        "_row_id": list(range(len(df))),
        "address": addr_series.tolist(),
    }
    if postcode_col:
        payload_dict["postcode_v"] = (
            df[postcode_col].fillna("").astype(str).str.strip().tolist()
        )
    if citycode_col:
        payload_dict["citycode_v"] = (
            df[citycode_col].fillna("").astype(str).str.strip().tolist()
        )
    payload = pd.DataFrame(payload_dict)

    csv_bytes = payload.to_csv(sep=",", encoding="utf-8", index=False).encode("utf-8")

    data: list[tuple[str, str]] = [("columns", "address")]
    if postcode_col:
        data.append(("postcode", "postcode_v"))
    if citycode_col:
        data.append(("citycode", "citycode_v"))

    files = {"data": ("input.csv", csv_bytes, "text/csv")}
    r = session.post(BAN_CSV_URL, data=data, files=files, timeout=BATCH_TIMEOUT)
    r.raise_for_status()

    result = pd.read_csv(io.BytesIO(r.content), dtype=str).fillna("")
    if "_row_id" not in result.columns:
        raise RuntimeError(
            "Réponse inattendue de l'API BAN : la colonne '_row_id' est absente."
        )
    result["_row_id"] = pd.to_numeric(result["_row_id"], errors="coerce").astype("Int64")
    result = result.sort_values("_row_id").reset_index(drop=True)

    out = pd.DataFrame({
        "adresse_envoyee": addr_series.values,
        "adresse_ban": result.get("result_label", "").astype(str),
        "lat": pd.to_numeric(result.get("latitude", ""), errors="coerce"),
        "lon": pd.to_numeric(result.get("longitude", ""), errors="coerce"),
        "score_ban": pd.to_numeric(result.get("result_score", ""), errors="coerce"),
    })
    return out


# ---------------------------------------------------------------------------
# Fallback cadastre — API Carto IGN
# ---------------------------------------------------------------------------

def lookup_commune_insee(
    postcode: str,
    city: str,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """Récupère le code INSEE d'une commune via la BAN (endpoint type=municipality).

    Renvoie le code INSEE (5 chiffres) ou None si introuvable.
    """
    if not (postcode or city):
        return None
    if session is None:
        session = requests.Session()
    q = (postcode + " " + city).strip() if postcode and city else (postcode or city).strip()
    params = {"q": q, "limit": 1, "type": "municipality"}
    try:
        r = session.get(BAN_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    feats = data.get("features") or []
    if not feats:
        return None
    return feats[0].get("properties", {}).get("citycode")


def lookup_parcelle_coords(
    code_insee: str,
    section: str,
    numero: str,
    session: Optional[requests.Session] = None,
) -> Optional[dict]:
    """Appelle l'API Carto Cadastre IGN et renvoie le centroïde de la parcelle.

    Renvoie {"lat": float, "lon": float} ou None si la parcelle n'existe pas
    ou que l'API échoue.
    """
    if not (code_insee and section and numero):
        return None
    if session is None:
        session = requests.Session()
    # L'API Carto attend la section sur 2 caractères (complétée par 0 si besoin)
    section_norm = section.zfill(2) if len(section) < 2 else section
    params = {
        "code_insee": code_insee,
        "section": section_norm,
        "numero": numero,
    }
    try:
        r = session.get(APICARTO_PARCELLE_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    features = data.get("features") or []
    if not features:
        return None
    geom = features[0].get("geometry")
    if not geom:
        return None
    try:
        poly = shape(geom)
        centroid = poly.centroid
        return {"lat": float(centroid.y), "lon": float(centroid.x)}
    except Exception:
        return None


def apply_cadastre_fallback(
    geo_df: pd.DataFrame,
    parcelles: pd.Series,
    postcodes: pd.Series,
    cities: pd.Series,
    score_threshold: float = FALLBACK_SCORE_THRESHOLD,
    session: Optional[requests.Session] = None,
    progress_cb=None,
) -> pd.DataFrame:
    """Pour les lignes avec un score BAN < seuil et une parcelle extraite,
    tente de retrouver les coordonnées via l'API Carto Cadastre.

    Entrées (alignées par index) :
        geo_df    : DataFrame retourné par geocode_batch_csv
                    (colonnes lat, lon, score_ban, adresse_envoyee, adresse_ban)
        parcelles : Series de dicts {prefixe, section, numero} ou None
        postcodes : Series de codes postaux (str)
        cities    : Series de villes (str)

    Renvoie un DataFrame copie de geo_df avec :
        - lat/lon remplacées si fallback réussi
        - colonne 'source_geocodage' ajoutée : "BAN", "Cadastre",
          "BAN (cadastre indisponible)" ou "" si pas de géocodage du tout
    """
    if session is None:
        session = requests.Session()

    out = geo_df.copy()
    out["source_geocodage"] = ""

    # Source par défaut selon le résultat BAN
    has_ban_result = out["lat"].notna()
    out.loc[has_ban_result, "source_geocodage"] = "BAN"

    # Candidats au fallback : score < seuil ET parcelle extraite
    score_num = pd.to_numeric(out["score_ban"], errors="coerce")
    need_fallback = (score_num < score_threshold) | out["lat"].isna()
    has_parcelle = parcelles.apply(lambda x: isinstance(x, dict))
    candidates = out.index[need_fallback & has_parcelle]

    if len(candidates) == 0:
        return out

    insee_cache: dict[tuple[str, str], Optional[str]] = {}
    done = 0
    total = len(candidates)
    for idx in candidates:
        cp = str(postcodes.loc[idx]).strip() if idx in postcodes.index else ""
        ville = str(cities.loc[idx]).strip() if idx in cities.index else ""
        parc = parcelles.loc[idx]

        cache_key = (cp, ville.lower())
        if cache_key not in insee_cache:
            insee_cache[cache_key] = lookup_commune_insee(cp, ville, session=session)
        insee = insee_cache[cache_key]

        coords = None
        if insee:
            coords = lookup_parcelle_coords(
                insee, parc["section"], parc["numero"], session=session
            )
        if coords is not None:
            out.at[idx, "lat"] = coords["lat"]
            out.at[idx, "lon"] = coords["lon"]
            out.at[idx, "source_geocodage"] = "Cadastre"
        elif out.at[idx, "source_geocodage"] == "BAN":
            out.at[idx, "source_geocodage"] = "BAN (cadastre indisponible)"

        done += 1
        if progress_cb is not None:
            progress_cb(done, total)
        time.sleep(SLEEP_BETWEEN_CADASTRE)

    return out


# ---------------------------------------------------------------------------
# Référentiel QPV
# ---------------------------------------------------------------------------

def _pick_shapefile_in_zip(namelist: list[str]) -> Optional[str]:
    """Choisit le meilleur shapefile dans un zip (hexagone + outre-mer en WGS84 si dispo)."""
    shps = [n for n in namelist if n.lower().endswith(".shp")]
    if not shps:
        return None
    if len(shps) == 1:
        return shps[0]

    def score(name: str) -> int:
        n = name.lower()
        s = 0
        if "outre_mer" in n or "outre-mer" in n:
            s += 10
        if "wgs84" in n:
            s += 8
        if "hexagonal" in n:
            s += 5
        if "france" in n:
            s += 2
        # Pénalise les fichiers spécifiques à un DOM seul
        for dom in ("guadeloupe", "guyane", "lareunion", "la_reunion", "martinique", "mayotte"):
            if dom in n:
                s -= 5
        return s

    return max(shps, key=score)


def load_qpv(qpv_path: str) -> gpd.GeoDataFrame:
    """Charge le référentiel QPV (shapefile, GeoJSON, GeoPackage, KML, zip).

    Pour un .zip contenant un dossier `SHP/` avec plusieurs shapefiles,
    sélectionne automatiquement la couche France hexagonale + Outre-Mer en WGS84.
    Reprojette automatiquement en WGS84 pour matcher les points BAN.
    """
    p = Path(qpv_path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            shp_inside = _pick_shapefile_in_zip(z.namelist())
        if shp_inside is None:
            # Pas de .shp dans le zip — geopandas peut peut-être lire (cas geojson zippé)
            gdf = gpd.read_file(str(p))
        else:
            # Syntaxe GDAL /vsizip/ pour lire le shapefile imbriqué
            gdf = gpd.read_file(f"/vsizip/{p.as_posix()}/{shp_inside}")
    else:
        gdf = gpd.read_file(str(p))

    if gdf.crs is None:
        # Par défaut, l'ANCT publie en Lambert-93 (EPSG:2154)
        gdf = gdf.set_crs(epsg=2154)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def detect_col(gdf: gpd.GeoDataFrame, candidates, override: str = "") -> Optional[str]:
    """Trouve dans gdf une colonne parmi candidates, ou utilise override si fourni."""
    if override:
        if override in gdf.columns:
            return override
        raise ValueError(
            f"La colonne '{override}' n'existe pas dans le référentiel QPV. "
            f"Colonnes disponibles : {list(gdf.columns)}"
        )
    cols_lower = {c.lower(): c for c in gdf.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def find_qpv(lat, lon, qpv: gpd.GeoDataFrame, code_col, name_col, sindex):
    """Point-in-polygon unitaire avec index spatial."""
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return False, None, None
    pt = Point(lon, lat)
    candidate_idx = list(sindex.intersection(pt.bounds))
    if not candidate_idx:
        return False, None, None
    candidates = qpv.iloc[candidate_idx]
    hit = candidates[candidates.contains(pt)]
    if hit.empty:
        return False, None, None
    row = hit.iloc[0]
    return (
        True,
        row.get(code_col) if code_col else None,
        row.get(name_col) if name_col else None,
    )


def find_qpv_batch(
    lat: pd.Series,
    lon: pd.Series,
    qpv: gpd.GeoDataFrame,
    code_col: Optional[str],
    name_col: Optional[str],
) -> pd.DataFrame:
    """Point-in-polygon vectorisé sur deux séries lat/lon.

    Renvoie un DataFrame indexé comme les séries d'entrée avec les colonnes :
        en_qpv (bool), code_qpv, nom_qpv
    """
    idx = lat.index
    out = pd.DataFrame(
        {
            "en_qpv": pd.Series(False, index=idx, dtype=bool),
            "code_qpv": pd.Series([None] * len(idx), index=idx, dtype=object),
            "nom_qpv": pd.Series([None] * len(idx), index=idx, dtype=object),
        }
    )
    valid = lat.notna() & lon.notna()
    if not valid.any():
        return out

    sub_lat = lat[valid]
    sub_lon = lon[valid]
    geometry = gpd.points_from_xy(sub_lon, sub_lat)
    pts = gpd.GeoDataFrame(geometry=geometry, index=sub_lat.index, crs="EPSG:4326")

    join_cols = ["geometry"]
    if code_col:
        join_cols.append(code_col)
    if name_col and name_col != code_col:
        join_cols.append(name_col)

    joined = gpd.sjoin(pts, qpv[join_cols], how="left", predicate="within")
    joined = joined.loc[~joined.index.duplicated(keep="first")]

    matched = joined["index_right"].notna()
    out.loc[matched[matched].index, "en_qpv"] = True
    if code_col and code_col in joined.columns:
        out.loc[joined.index, "code_qpv"] = joined[code_col].values
    if name_col and name_col in joined.columns:
        out.loc[joined.index, "nom_qpv"] = joined[name_col].values

    return out


# ---------------------------------------------------------------------------
# Téléchargement auto du référentiel QPV depuis data.gouv.fr
# ---------------------------------------------------------------------------

def get_qpv_resources(session: Optional[requests.Session] = None) -> list[dict]:
    """Liste les ressources du dataset QPV publiées sur data.gouv.fr.

    Renvoie une liste de dicts avec les champs : title, format, url, filesize.
    """
    if session is None:
        session = requests.Session()
    r = session.get(QPV_DATASET_API, timeout=15)
    r.raise_for_status()
    data = r.json()
    raw_resources = data.get("resources")
    if not isinstance(raw_resources, list):
        raise RuntimeError(
            "Réponse inattendue de l'API data.gouv.fr : "
            "le champ 'resources' n'est pas une liste."
        )
    out = []
    for res in raw_resources:
        if not isinstance(res, dict):
            continue
        out.append({
            "title": res.get("title") or "",
            "format": (res.get("format") or "").lower(),
            "url": res.get("url") or "",
            "filesize": res.get("filesize") or 0,
        })
    return out


def pick_shapefile_resource(resources: list[dict]) -> Optional[dict]:
    """Choisit la ressource shapefile zip QPV 2024 (préfère SHP-seul à un pack combiné)."""
    def score(res: dict) -> int:
        title = res["title"].lower()
        url = res["url"].lower()
        if not url.endswith(".zip"):
            return 0
        s = 10  # base : c'est un zip
        is_2024 = "2024" in title or "2024" in url
        is_2015 = "2015" in title or "2015" in url
        if is_2024:
            s += 20
        if is_2015 and not is_2024:
            s -= 50  # on évite le millésime obsolète
        has_shp = "shp" in title or "shapefile" in title
        has_geojson = "geojson" in title or "json" in title.split()
        has_gpkg = "gpkg" in title
        if has_shp and not has_geojson and not has_gpkg:
            s += 15  # SHP pur, idéal
        elif has_shp:
            s += 5   # pack combiné
        return s
    ranked = sorted(resources, key=score, reverse=True)
    if not ranked or score(ranked[0]) < 10:
        return None
    return ranked[0]


def download_qpv_dataset(
    dest_dir: Path,
    session: Optional[requests.Session] = None,
    progress_cb=None,
) -> Path:
    """Télécharge le shapefile QPV depuis data.gouv.fr dans `dest_dir`.

    Renvoie le chemin du fichier .zip téléchargé. Si déjà présent, ne re-télécharge pas.
    `progress_cb(downloaded, total)` est appelé pendant le téléchargement si fourni.
    """
    if session is None:
        session = requests.Session()
    dest_dir.mkdir(parents=True, exist_ok=True)

    resources = get_qpv_resources(session)
    res = pick_shapefile_resource(resources)
    if res is None:
        raise RuntimeError(
            "Impossible de trouver la ressource shapefile QPV sur data.gouv.fr. "
            "Télécharge-la manuellement depuis "
            "https://www.data.gouv.fr/datasets/" + QPV_DATASET_SLUG
        )

    url = res["url"]
    filename = url.rsplit("/", 1)[-1] or "qpv.zip"
    if not filename.endswith(".zip"):
        filename += ".zip"
    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return target

    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or res["filesize"] or 0)
        done = 0
        tmp = target.with_suffix(".zip.part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress_cb is not None and total:
                    progress_cb(done, total)
        tmp.replace(target)
    return target


# ---------------------------------------------------------------------------
# Lecture CSV (auto-détection du séparateur)
# ---------------------------------------------------------------------------

def read_csv_auto(path: str, sep_override: str = "", encoding: str = "utf-8-sig") -> pd.DataFrame:
    """Lit le CSV en détectant automatiquement le séparateur ; / , si besoin."""
    if sep_override:
        return pd.read_csv(path, sep=sep_override, encoding=encoding, dtype=str).fillna("")
    with open(path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(4096)
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding=encoding, dtype=str).fillna("")


# ---------------------------------------------------------------------------
# Programme principal (CLI)
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="CSV d'entrée")
    p.add_argument("--output", required=True, help="CSV de sortie enrichi")
    p.add_argument("--qpv", required=True, help="Fichier QPV (.shp, .geojson, .gpkg, .zip)")

    p.add_argument("--batch", action="store_true",
                   help="Utilise l'API BAN batch (beaucoup plus rapide)")

    p.add_argument("--col-numero", default="Numero",       help="Colonne numéro de voie")
    p.add_argument("--col-rue",    default="Rue",          help="Colonne nom de voie")
    p.add_argument("--col-cp",     default="Code postal",  help="Colonne code postal")
    p.add_argument("--col-ville",  default="Ville",        help="Colonne ville")

    p.add_argument("--sep",      default="",          help="Séparateur CSV (auto si vide)")
    p.add_argument("--encoding", default="utf-8-sig", help="Encodage du CSV (utf-8-sig, cp1252...)")

    p.add_argument("--qpv-code-col", default="", help="Forcer le nom de la colonne code dans le QPV")
    p.add_argument("--qpv-name-col", default="", help="Forcer le nom de la colonne nom dans le QPV")

    args = p.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"ERREUR : fichier d'entrée introuvable : {args.input}")
    if not Path(args.qpv).exists():
        sys.exit(f"ERREUR : référentiel QPV introuvable : {args.qpv}")

    print("[1/3] Référentiel QPV")
    qpv = load_qpv(args.qpv)
    print(f"  {len(qpv)} polygones chargés.")
    code_col = detect_col(qpv, QPV_CODE_CANDIDATES, args.qpv_code_col)
    name_col = detect_col(qpv, QPV_NAME_CANDIDATES, args.qpv_name_col)
    print(f"  Colonne code QPV : {code_col or '(non trouvée — vide en sortie)'}")
    print(f"  Colonne nom  QPV : {name_col or '(non trouvée — vide en sortie)'}")

    print(f"\n[2/3] Lecture du CSV {args.input}")
    df = read_csv_auto(args.input, args.sep, args.encoding)
    print(f"  {len(df)} ligne(s) à traiter.")
    missing = [c for c in [args.col_numero, args.col_rue, args.col_cp, args.col_ville]
               if c not in df.columns]
    if missing:
        sys.exit(
            f"ERREUR : colonnes manquantes dans le CSV : {missing}\n"
            f"  Colonnes disponibles : {list(df.columns)}"
        )

    print(f"\n[3/3] Géocodage BAN + test QPV ({'batch' if args.batch else 'unitaire'})")
    session = requests.Session()
    addr_cols = [args.col_numero, args.col_rue, args.col_cp, args.col_ville]

    if args.batch:
        geo_df = geocode_batch_csv(
            df, address_cols=addr_cols,
            postcode_col=args.col_cp, citycode_col="",
            session=session,
        )
        qpv_df = find_qpv_batch(geo_df["lat"], geo_df["lon"], qpv, code_col, name_col)

        enriched = df.copy()
        enriched["adresse_envoyee"] = geo_df["adresse_envoyee"].values
        enriched["adresse_ban"]     = geo_df["adresse_ban"].values
        enriched["lat"]             = geo_df["lat"].values
        enriched["lon"]             = geo_df["lon"].values
        enriched["score_ban"]       = geo_df["score_ban"].values
        enriched["en_qpv"] = [
            ("Adresse non géocodée" if pd.isna(la) else ("Oui" if oui else "Non"))
            for la, oui in zip(geo_df["lat"], qpv_df["en_qpv"])
        ]
        enriched["code_qpv"] = qpv_df["code_qpv"].values
        enriched["nom_qpv"]  = qpv_df["nom_qpv"].values
    else:
        sindex = qpv.sindex
        out_rows = []
        n = len(df)
        for i, row in df.iterrows():
            numero = str(row[args.col_numero]).strip()
            rue    = str(row[args.col_rue]).strip()
            cp     = str(row[args.col_cp]).strip()
            ville  = str(row[args.col_ville]).strip()
            addr   = normalize_address(" ".join(x for x in [numero, rue, cp, ville] if x))

            geo = geocode(addr, postcode=cp, citycode="", session=session)
            en_qpv, code_q, nom_q = find_qpv(
                geo["lat"], geo["lon"], qpv, code_col, name_col, sindex
            )
            if geo["lat"] is None:
                statut = "Adresse non géocodée"
            elif en_qpv:
                statut = "Oui"
            else:
                statut = "Non"
            out_rows.append({
                "adresse_envoyee": addr,
                "adresse_ban":     geo["label"],
                "lat":             geo["lat"],
                "lon":             geo["lon"],
                "score_ban":       geo["score"],
                "en_qpv":          statut,
                "code_qpv":        code_q,
                "nom_qpv":         nom_q,
            })
            if (i + 1) % 10 == 0 or (i + 1) == n:
                print(f"  {i+1}/{n} ligne(s) traitée(s)…")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        enriched = pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1
        )

    out_sep = args.sep or ";"
    enriched.to_csv(args.output, sep=out_sep, encoding="utf-8-sig", index=False)
    print(f"\n✓ CSV enrichi écrit : {args.output}")

    nb_oui = int((enriched["en_qpv"] == "Oui").sum())
    nb_non = int((enriched["en_qpv"] == "Non").sum())
    nb_err = int((enriched["en_qpv"] == "Adresse non géocodée").sum())
    score_num = pd.to_numeric(enriched["score_ban"], errors="coerce")
    nb_low = int((score_num < 0.7).sum())
    print("\n── Résumé ───────────────────────────────")
    print(f"  En QPV              : {nb_oui}")
    print(f"  Hors QPV            : {nb_non}")
    print(f"  Non géocodées       : {nb_err}")
    print(f"  Score BAN < 0.7     : {nb_low}  (à revérifier manuellement)")
    print("─────────────────────────────────────────")


if __name__ == "__main__":
    main()
