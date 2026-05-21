# Vérification QPV

Outil pour vérifier si une adresse française se situe dans un **Quartier Prioritaire de la Politique de la Ville (QPV)** du référentiel ANCT 2024.

- **Interface web** ([Streamlit](https://streamlit.io)) — adresse unitaire ou lot CSV, carte interactive
- **Script CLI** — pour intégrer dans un pipeline de traitement

Le géocodage utilise l'API publique gratuite de la BAN ([adresse.data.gouv.fr](https://adresse.data.gouv.fr)). Le test d'appartenance est un point-dans-polygone sur le référentiel officiel ANCT publié sur [data.gouv.fr](https://www.data.gouv.fr/datasets/quartiers-prioritaires-de-la-politique-de-la-ville-qpv).

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Lancer l'application Streamlit](#lancer-lapplication-streamlit)
- [Utilisation en ligne de commande](#utilisation-en-ligne-de-commande)
- [Format de sortie](#format-de-sortie)
- [Tests](#tests)
- [Notes & limites](#notes--limites)
- [Licence](#licence)

## Fonctionnalités

- 📥 Téléchargement automatique du référentiel QPV depuis data.gouv.fr (ou upload manuel)
- 🔍 Vérification d'une adresse unitaire avec carte centrée sur le point
- 📂 Traitement d'un lot CSV avec **mode batch** (un seul appel à l'API BAN, ~50× plus rapide que le mode unitaire)
- 🗺️ Carte interactive (pydeck) des résultats avec polygones QPV visibles
- 💾 Cache local du référentiel et des géocodages unitaires (TTL 24 h)
- 📋 Bouton « Exemple » qui charge un CSV de démonstration

## Installation

Python 3.10 ou plus récent.

```bash
git clone https://github.com/<ton-user>/verif-qpv.git
cd verif-qpv
pip install -r requirements.txt
```

Sur Windows, si l'installation de `geopandas` / `pyogrio` échoue, mets à jour `pip` puis réessaie :
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Lancer l'application Streamlit

```bash
streamlit run app.py
```

L'app ouvre `http://localhost:8501`.

### Premier lancement

1. Dans la barre latérale, choisis le mode **📥 Télécharger depuis data.gouv.fr** et clique sur le bouton de téléchargement.
   - Le fichier (~50 Mo) est mis en cache dans `~/.cache/verif-qpv/` ; pas de re-téléchargement aux lancements suivants.
2. Onglet **🔍 Adresse unitaire** ou **📂 Lot CSV** pour vérifier.

### Alternatives pour fournir le référentiel

| Mode | Quand utiliser |
|---|---|
| 📥 Télécharger depuis data.gouv.fr | Premier usage, machine connectée à Internet |
| 📂 Uploader un fichier | Machine isolée ; envoyer un `.zip`, `.geojson` ou `.gpkg` |
| 📁 Chemin local | Le fichier est déjà sur disque (par ex. `C:/data/qpv/...shp`) |

## Utilisation en ligne de commande

```bash
# Mode batch (recommandé) : ~50× plus rapide
python verif_qpv.py \
    --input dossiers.csv \
    --output enrichi.csv \
    --qpv QP2024_France_hexagonale.zip \
    --batch

# Mode unitaire (legacy)
python verif_qpv.py \
    --input dossiers.csv \
    --output enrichi.csv \
    --qpv QP2024_France_hexagonale.shp \
    --col-numero "N°" \
    --col-rue "Voie" \
    --col-cp "CP" \
    --col-ville "Commune"
```

Les noms de colonnes par défaut (`Numero`, `Rue`, `Code postal`, `Ville`) peuvent être surchargés via les options `--col-*`. L'encodage et le séparateur sont auto-détectés.

## Format de sortie

Le CSV enrichi reprend toutes les colonnes d'entrée et ajoute :

| Colonne | Description |
|---|---|
| `adresse_envoyee` | Adresse reconstituée envoyée à la BAN |
| `adresse_ban` | Libellé normalisé renvoyé par la BAN |
| `lat`, `lon` | Coordonnées WGS84 |
| `score_ban` | Confiance du géocodage (0 à 1, > 0.7 = fiable) |
| `en_qpv` | `Oui` / `Non` / `Adresse non géocodée` |
| `code_qpv` | Code officiel du QPV (ex. `QN075033`) |
| `nom_qpv` | Nom du quartier prioritaire |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Les tests couvrent les fonctions cœur (`find_qpv`, `find_qpv_batch`, `geocode`, `geocode_batch_csv`, `detect_col`, `pick_shapefile_resource`) sans appel réseau (API BAN mockée).

## Notes & limites

- **Score BAN < 0.7** : à revérifier manuellement, le géocodage peut être approximatif (rues homonymes, adresses ambiguës).
- **Frontières** : un point exactement sur la limite d'un polygone QPV est considéré comme dedans (`contains` géométrique).
- **Référentiel** : seul le QPV 2024 hexagonal est ciblé par défaut. Pour les DOM, télécharge la ressource correspondante sur data.gouv.fr et utilise le mode upload.
- **API BAN** : pas d'authentification, ~50 req/s en unitaire, 50 Mo max par requête batch. Aucune adresse n'est envoyée à un service tiers en dehors d'`adresse.data.gouv.fr`.
- **Cache** : le référentiel est mis en cache via `@st.cache_resource` (clé = chemin / hash), les géocodages unitaires via `@st.cache_data` (TTL 24 h).

## Sources

- API BAN — [adresse.data.gouv.fr](https://adresse.data.gouv.fr)
- Référentiel QPV — [data.gouv.fr](https://www.data.gouv.fr/datasets/quartiers-prioritaires-de-la-politique-de-la-ville-qpv)

## Auteur

Réalisé par **medidev34**.

## Licence

[MIT](LICENSE) — © 2026 medidev34
