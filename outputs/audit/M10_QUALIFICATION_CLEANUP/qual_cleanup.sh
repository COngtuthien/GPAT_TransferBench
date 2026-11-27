#!/bin/bash
# M10 qualification-directory cleanup: inventory + preserve + exact-path delete.
# usage: qual_cleanup.sh inventory | delete
set -euo pipefail
Q="$HOME/workdir/m10_qualification"
OUT="$HOME/workdir/m10_qual_cleanup"
T1="$Q/243ac81aa53b9c6c686a323c73d27522f3c59dd0"
T2="$Q/a4dd841b4b2a3e636af3bad2818a0effdc87a544"
check() {
  for t in "$T1" "$T2"; do
    [ -d "$t" ] && [ ! -L "$t" ] || { echo "REFUSE not a real dir: $t"; exit 9; }
    [ "$(readlink -f "$t")" = "$t" ] || { echo "REFUSE non-canonical: $t"; exit 9; }
    [ "$(dirname "$t")" = "$Q" ] || { echo "REFUSE outside root: $t"; exit 9; }
    [ -z "$(find "$t" -type l -print -quit)" ] || { echo "REFUSE symlink inside $t"; exit 9; }
    [ -z "$(find "$t" ! -user "$(id -un)" -print -quit)" ] || { echo "REFUSE foreign owner in $t"; exit 9; }
    [ -z "$(find "$t" -xdev ! -type f ! -type d -print -quit)" ] || { echo "REFUSE special file in $t"; exit 9; }
    [ "$(stat -c %d "$t")" = "$(stat -c %d "$Q")" ] || { echo "REFUSE mount crossing $t"; exit 9; }
  done
  if ps -eo args | grep -q "[r]un_m10_detector\|[m]10_launcher_qualification\|[m]10_run_auditor"; then
    echo "REFUSE M10 process running"; exit 9; fi
  if command -v lsof >/dev/null && lsof +D "$Q" 2>/dev/null | grep -q .; then echo "REFUSE open files"; exit 9; fi
}
case "$1" in
inventory)
  check
  mkdir -p "$OUT"
  for t in "$T1" "$T2"; do
    b=$(basename "$t")
    (cd "$Q" && find "$b" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$OUT/inventory_$b.sha256"
    (cd "$Q" && find "$b" -type f -printf "%s\t%p\n" | LC_ALL=C sort -k2) > "$OUT/sizes_$b.tsv"
  done
  (cd "$Q" && find 243ac81aa53b9c6c686a323c73d27522f3c59dd0 a4dd841b4b2a3e636af3bad2818a0effdc87a544 -type f \
      ! -name '*.pt' ! -name '*.png' -print0 | LC_ALL=C sort -z \
      | tar --null -T - --owner=0 --group=0 --numeric-owner --mtime='2026-11-26 00:00Z' -czf "$OUT/qualification_raw_no_pt_no_png.tgz")
  # verify archive: every archived file hash equals the inventory
  V=$(mktemp -d "$OUT/verify.XXXX"); tar -xzf "$OUT/qualification_raw_no_pt_no_png.tgz" -C "$V"
  (cd "$V" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sed 's#  \./#  #') > "$OUT/archive_check.sha256"
  rm -r -- "$V"
  cat "$OUT"/inventory_*.sha256 | grep -v -E '\.(pt|png)$' | LC_ALL=C sort -k2 > "$OUT/expected_archived.sha256"
  LC_ALL=C sort -k2 "$OUT/archive_check.sha256" | cmp - "$OUT/expected_archived.sha256" && echo ARCHIVE_VERIFIED
  wc -l "$OUT"/*.sha256; du -sb "$T1" "$T2"; ls -l "$OUT"
  ;;
delete)
  check
  [ -f "$OUT/inventory_$(basename "$T1").sha256" ] && [ -f "$OUT/inventory_$(basename "$T2").sha256" ] || { echo "REFUSE no inventory"; exit 9; }
  for t in "$T1" "$T2"; do
    b=$(basename "$t")
    n=$(find "$t" -type f | wc -l); m=$(wc -l < "$OUT/inventory_$b.sha256")
    [ "$n" = "$m" ] || { echo "REFUSE inventory drift $b $n vs $m"; exit 9; }
  done
  df -B1 --output=avail "$Q" | tail -1 > "$OUT/avail_before"
  rm -r --one-file-system -- "$T1" "$T2"
  df -B1 --output=avail "$Q" | tail -1 > "$OUT/avail_after"
  date -u +%FT%TZ > "$OUT/deleted_utc"
  ls -la "$Q"; cat "$OUT/avail_before" "$OUT/avail_after" "$OUT/deleted_utc"
  ;;
esac
