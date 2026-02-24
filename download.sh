#!/usr/bin/env bash
set -euo pipefail

# Batch download shared Google Drive files and optionally unzip .zip outputs.
# List format (one per line):
#   <gdrive_share_url> <output_filename>
#
# Usage:
#   ./gdrive_batch_download.sh list.txt [output_dir]
#
# Notes:
# - If output filename ends with .zip (case-insensitive), it will unzip into output_dir
#   and delete the zip on success.

LIST_FILE="${1:-}"
OUT_DIR="${2:-.}"

if [[ -z "${LIST_FILE}" || ! -f "${LIST_FILE}" ]]; then
  echo "Usage: $0 <list.txt> [output_dir]" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

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

html_attr_value() {
  local text="$1" attr="$2"
  printf '%s' "$text" | sed -n "s/.*${attr}=\"\([^\"]*\)\".*/\1/p" | head -n 1
}

html_input_value() {
  local file="$1" name="$2"
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

  # First request: cookies + either direct download or interstitial HTML
  curl -fsSL -D "$headers" -c "$cookie" \
    "https://drive.google.com/uc?export=download&id=${file_id}" \
    -o "$page"

  # If interstitial contains a download-form, use its action + hidden inputs.
  local formline action confirm uuid at
  formline="$(grep -m1 'id="download-form"' "$page" || true)"
  action=""
  if [[ -n "$formline" ]]; then
    action="$(html_attr_value "$formline" "action" || true)"
  fi

  if [[ -n "$action" ]]; then
    confirm="$(html_input_value "$page" "confirm" || true)"
    uuid="$(html_input_value "$page" "uuid" || true)"
    at="$(html_input_value "$page" "at" || true)"  # sometimes present

    if [[ -z "$confirm" || -z "$uuid" ]]; then
      echo "ERROR: Could not extract confirm/uuid for: $url" >&2
      return 3
    fi

    local dl="${action}?id=${file_id}&export=download&confirm=${confirm}&uuid=${uuid}"
    [[ -n "$at" ]] && dl="${dl}&at=${at}"

    echo "Downloading -> $out"
    curl -fL -b "$cookie" -c "$cookie" \
      -H "Referer: https://drive.google.com/" \
      "$dl" -o "$out"
  else
    # Fallback to direct endpoint
    echo "Downloading -> $out"
    curl -fL -b "$cookie" -c "$cookie" \
      "https://drive.google.com/uc?export=download&id=${file_id}" \
      -o "$out"
  fi

  if is_html_file "$out"; then
    echo "ERROR: Downloaded HTML instead of file for URL: $url" >&2
    echo "       Output: $out" >&2
    echo "       Hint: file may require permission/login, or Google changed the flow." >&2
    return 4
  fi
}

maybe_unzip_and_cleanup() {
  local filepath="$1"
  local outdir="$2"

  local filename ext
  filename="$(basename -- "$filepath")"
  ext="${filename##*.}"
  ext="$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"

  if [[ "$ext" == "zip" ]]; then
    if ! command -v unzip >/dev/null 2>&1; then
      echo "ERROR: unzip is not installed, cannot unpack: $filepath" >&2
      return 5
    fi
    echo "Unpacking $filename"
    # -q quiet, -o overwrite, -d target dir
    unzip -q -o "$filepath" -d "$outdir"
    echo "Unpacked OK, removing $filename"
    rm -f -- "$filepath"
  fi
}

# Read list file
while IFS= read -r line || [[ -n "$line" ]]; do
  # trim whitespace
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

  # Allow quoted filenames in list file
  # shellcheck disable=SC2086
  out_name="$(eval "printf '%s' $rest")"
  out_path="$OUT_DIR/$out_name"

  download_gdrive_one "$url" "$out_path"
  maybe_unzip_and_cleanup "$out_path" "$OUT_DIR"
done < "$LIST_FILE"

echo "All done."
