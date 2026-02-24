#!/usr/bin/env bash
set -euo pipefail

# Batch download shared Google Drive files.
# List format (one per line):
#   <gdrive_share_url> <output_filename>
#
# Usage:
#   ./gdrive_batch_download.sh list.txt [output_dir]

LIST_FILE="${1:-}"
OUT_DIR="${2:-.}"

if [[ -z "${LIST_FILE}" || ! -f "${LIST_FILE}" ]]; then
  echo "Usage: $0 <list.txt> [output_dir]" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Extract file id from common Google Drive URL formats
extract_file_id() {
  local url="$1"
  if [[ "$url" =~ /file/d/([^/]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  elif [[ "$url" =~ [\?\&]id=([^&]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    return 1
  fi
}

# Extract an attribute value from an HTML line (very small/portable parser)
# usage: html_attr_value "<html...>" attrname
html_attr_value() {
  local text="$1" attr="$2"
  # matches attr="VALUE"
  printf '%s' "$text" | sed -n "s/.*${attr}=\"\([^\"]*\)\".*/\1/p" | head -n 1
}

# Extract hidden input value by name="X" value="Y"
# usage: html_input_value file.html input_name
html_input_value() {
  local file="$1" name="$2"
  # Grab the first <input ... name="NAME" ... value="...">
  local line
  line="$(grep -m1 "name=\"$name\"" "$file" || true)"
  [[ -z "$line" ]] && return 1
  html_attr_value "$line" "value"
}

is_html_file() {
  local path="$1"
  if command -v file >/dev/null 2>&1; then
    [[ "$(file -b --mime-type "$path" 2>/dev/null || true)" == "text/html" ]]
  else
    head -c 256 "$path" | grep -qiE '<!doctype html|<html'
  fi
}

download_gdrive_one() {
  local url="$1"
  local out="$2"

  local file_id
  if ! file_id="$(extract_file_id "$url")"; then
    echo "ERROR: Could not parse file id from: $url" >&2
    return 2
  fi

  local tmpdir cookie page headers
  tmpdir="$(mktemp -d)"
  cookie="$tmpdir/cookie.txt"
  page="$tmpdir/page.html"
  headers="$tmpdir/headers.txt"
  trap 'rm -rf "$tmpdir"' RETURN

  # First hit: capture cookies and either get file directly or get the warning HTML
  curl -fsSL -D "$headers" -c "$cookie" \
    "https://drive.google.com/uc?export=download&id=${file_id}" \
    -o "$page"

  # If Google already sent a file (rare but possible), just re-download to out
  # Otherwise, parse the download form action + hidden fields.
  local action confirm uuid at
  action="$(grep -m1 'id="download-form"' "$page" | sed -n 's/.*action="\([^"]*\)".*/\1/p' | head -n 1 || true)"

  if [[ -n "$action" ]]; then
    confirm="$(html_input_value "$page" "confirm" || true)"
    uuid="$(html_input_value "$page" "uuid" || true)"
    at="$(html_input_value "$page" "at" || true)"  # sometimes present

    # Sanity
    if [[ -z "$confirm" || -z "$uuid" ]]; then
      echo "ERROR: Could not extract confirm/uuid for: $url" >&2
      return 3
    fi

    # Build query
    local dl="${action}?id=${file_id}&export=download&confirm=${confirm}&uuid=${uuid}"
    if [[ -n "$at" ]]; then
      dl="${dl}&at=${at}"
    fi

    echo "Downloading -> $out"
    curl -fL -b "$cookie" -c "$cookie" \
      -H "Referer: https://drive.google.com/" \
      "$dl" -o "$out"

  else
    # No download form found: try direct download endpoint
    echo "Downloading -> $out"
    curl -fL -b "$cookie" -c "$cookie" \
      "https://drive.google.com/uc?export=download&id=${file_id}" \
      -o "$out"
  fi

  if is_html_file "$out"; then
    echo "ERROR: Downloaded HTML instead of file for URL: $url" >&2
    echo "       Output: $out" >&2
    echo "       Hint: file may require permission/login, or Google changed the interstitial again." >&2
    return 4
  fi
}

# Read list file
while IFS= read -r line || [[ -n "$line" ]]; do
  # trim
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  url="$(printf '%s\n' "$line" | awk '{print $1}')"
  rest="$(printf '%s\n' "$line" | cut -d' ' -f2-)"
  if [[ -z "$rest" ]]; then
    echo "ERROR: Missing output filename in line: $line" >&2
    exit 2
  fi

  # Allow quoted filenames
  # shellcheck disable=SC2086
  out_name="$(eval "printf '%s' $rest")"

  download_gdrive_one "$url" "$OUT_DIR/$out_name"
done < "$LIST_FILE"

echo "All done."
