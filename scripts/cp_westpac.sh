SRC=../evaluation_data/synthetic_transaction_linking
DST=${SRC}_westpac
mkdir -p "$DST"

for f in "$SRC"/*westpac*.png; do
  case=${f##*/}; case=${case%%_*}          # CASE003_westpac_standard.png -> CASE003
  /bin/cp -f "$SRC/${case}"_*.png "$DST/"  # copy every file for that case
done


echo "files: $(/bin/ls "$DST" | wc -l) | westpac stmts: $(/bin/ls "$DST"/*westpac*.png | wc -l) | cases: $(/bin/ls "$DST"/*westpac*.png | sed -E 's#.*/(CASE[0-9]+)_.*#\1#' | sort -u | wc -l)"