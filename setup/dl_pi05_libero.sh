#!/bin/bash
# Download gs://openpi-assets/checkpoints/pi05_libero via public HTTPS JSON API through proxy.
set -euo pipefail
source ~/webvla-env.sh

DEST="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero"
rm -rf "$DEST" "$DEST.partial" "$DEST.lock"
mkdir -p "$DEST"

PREFIX="checkpoints/pi05_libero/"
TOKEN=""
COUNT=0
while :; do
  URL="https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=${PREFIX}&fields=items(name,size),nextPageToken&pageSize=1000"
  if [ -n "$TOKEN" ]; then URL="$URL&pageToken=$TOKEN"; fi
  RESP=$(curl -sf "$URL")
  TOKEN=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('nextPageToken') or '')")
  echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for it in d.get('items', []):
    print(it['name'])
" > /tmp/obj_list.txt
  while read -r obj; do
    rel="${obj#$PREFIX}"
    [ -z "$rel" ] && continue
    case "$rel" in */) continue;; esac
    out="$DEST/$rel"
    mkdir -p "$(dirname "$out")"
    enc=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$obj")
    curl -sf --retry 3 -o "$out" "https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/$enc?alt=media"
    COUNT=$((COUNT+1))
  done < /tmp/obj_list.txt
  [ -z "$TOKEN" ] && break
done
echo "Downloaded $COUNT files:"
du -sh "$DEST"
find "$DEST" -type f | sort
