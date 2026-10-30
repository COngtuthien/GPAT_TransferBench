# read-only: path/size/sha256 and path/size/mtime listings of every completed run root
cd /home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6 || exit 1
date -u +START=%FT%TZ
for d in E06c/seed_42 E06c/seed_1337 E06c/seed_2026 E06b/seed_42 E06b/seed_1337 E06b/seed_2026 E05/seed_42 E05/seed_1337 E05/seed_2026 E03/seed_42 E03/seed_1337 E03/seed_2026; do
  ( cd $d && find . -type f -printf "%P %s %T@\n" | LC_ALL=C sort > /dev/fd/3 ) 3>&1 | sed "s#^#MTIME $d #"
  ( cd $d && find . -type f -printf "%P\n" | LC_ALL=C sort | while IFS= read -r f; do printf '%s %s %s\n' "$f" "$(stat -c %s "$f")" "$(nice -n 19 ionice -c3 sha256sum < "$f" | cut -d" " -f1)"; done ) | sed "s#^#SHA $d #"
done
date -u +END=%FT%TZ
