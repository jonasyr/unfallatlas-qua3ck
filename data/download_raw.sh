#!/usr/bin/env bash
# Downloads raw Unfallatlas CSV files from Destatis / Unfallatlas API.
# Output: data/raw/unfalldaten/unfalldaten_<AGS>.csv  (one file per Kreis)
#
# Source: https://unfallatlas.statistikportal.de/
# Years covered: 2016–2024
#
# TODO: implement download logic (e.g. via the Unfallatlas GeoServer WFS endpoint)
set -euo pipefail

mkdir -p "$(dirname "$0")/raw/unfalldaten"
echo "Download not yet automated. Obtain CSVs manually from https://unfallatlas.statistikportal.de/" >&2
exit 1
