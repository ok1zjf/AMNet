#!/usr/bin/env bash
set -euo pipefail

# Download Google Drive shared files from a list:
# Each non-empty, non-comment line must be:
#   <gdrive_share_url> <output_filename>
#
# Usage:
#   ./gdrive_batch_download.sh list.txt [output_dir]
#
# Example list.txt:
#   https://drive.google.com/file/d/FILEID/view?usp=sharing out.zip

LIST_FILE="${1:-}"
OUT_DIR="${2:-.}"

if [[ -z "${LIST_FILE}" || ! -f "${LIST_FILE}" ]]; then
  echo "Usage: $0 <list.txt> [output_dir]" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

download_gdrive() {
  local url="$1"
  local out="$2"

  # Extract file id from various URL styles
  local file_id=""
  if [[ "$url" =~ /file/d/([^/]+) ]]; then
    file_id="${BASH_REMATCH[1]}"
  elif [[ "$url" =~ [\?\&]id=([^&]+) ]]; then
    file_id="${BASH_REMATCH[1]}"
  else
    echo "ERROR: Could not parse file id from URL: $url" >&2
    return 2
  fi

  local tmpdir cookie html confirm final_url
  tmpdir="$(mktemp -d)"
  cookie="$tmpdir/cookie.txt"
  html="$tmpdir/confirm.html"
  trap 'rm -rf "$tmpdir"' RETURN

  # First request: capture cookies and possible confirm token page
  curl -sSL -c "$cookie" "https://drive.google.com/uc?export=download&id=${file_id}" -o "$html"

  # Extract confirm token if present (large file / virus scan interstitial)
  confirm="$(sed -n 's/.*confirm=\([0-9A-Za-z_]\+\).*/\1/p' "$html" | head -n 1)"

  if [[ -n "$confirm" ]]; then
    final_url="https://drive.google.com/uc?export=download&confirm=${confirm}&id=${file_id}"
  else
    final_url="https://drive.google.com/uc?export=download&id=${file_id}"
  fi

  echo "Downloading -> $out"
  curl -fL -b "$cookie" "$final_url" -o "$out"

  # Basic sanity check: Google sometimes returns an HTML page on failure
  if head -c 256 "$out" | grep -qiE '<!doctype html|<html'; then
    echo "ERROR: Downloaded HTML instead of file for URL: $url" >&2
    echo "       Output: $out" >&2
    return 3
  fi
}

# Read list file line by line
# Supports filenames with spaces if you quote them in the list file:
#   <url> "my file name.zip"
while IFS= read -r line || [[ -n "$line" ]]; do
  # Trim leading/trailing whitespace
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"

  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  # Parse: first field = URL, rest = filename (supports quotes via eval)
  # shellcheck disable=SC2086
  url="$(printf '%s\n' "$line" | awk '{print $1}')"
  rest="$(printf '%s\n' "$line" | cut -d' ' -f2-)"
  if [[ -z "$rest" ]]; then
    echo "ERROR: Missing output filename in line: $line" >&2
    exit 2
  fi

  # Handle optional quotes in filename
  # shellcheck disable=SC2086,SC2090
  out_name="$(eval "printf '%s' $rest")"

  download_gdrive "$url" "$OUT_DIR/$out_name"
done < "$LIST_FILE"

echo "All done."
