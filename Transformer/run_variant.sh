#!/usr/bin/env bash
# Train one (activation, width) variant of the substituted-FFN TinyBERT4 from
# scratch: Stage 1 + Stage 2 per layer, then one Stage-3 KD substitution over all
# the layers together.
#
#   ACT=bern|gelu  HIDDEN=312|600  LAYERS="0 1 2 3"  MODE=full|smoke \
#       bash Transformer/run_variant.sh
#
# MODE=full  (default) the recipe the shipped TABLE XII weights were trained with.
# MODE=smoke a few epochs per stage -- proves the pipeline runs end to end in
#            ~20 min on one GPU. It does NOT reproduce the paper accuracies.
#
# Output goes to Transformer/scratch/<tag>/ so the shipped models/ are never
# overwritten. Stages are skipped if their checkpoint already exists, so an
# interrupted run resumes; delete the scratch dir to start over.
#
# NOTE: retraining does not reproduce the published numbers digit-for-digit.
# Exact reproduction of TABLE XII comes from the shipped weights -- see
# `python Transformer/eval_release.py`.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
ACT=${ACT:?set ACT=bern or gelu}
HIDDEN=${HIDDEN:?set HIDDEN=312 or 600}
LAYERS=${LAYERS:-0 1 2 3}
MODE=${MODE:-full}
DEG=15
TEACHER=${TEACHER:-models/teacher_gelu_9037.pt}

if [ "$MODE" = smoke ]; then
  S1_EPOCHS=6; S1_RECAL=2; S1_FREEZE=4
  S2_EPOCHS=4; S2_RECAL=2; S2_FREEZE=2
  S3_EPOCHS=2
else
  S1_EPOCHS=100; S1_RECAL=10; S1_FREEZE=30
  S2_EPOCHS=30;  S2_RECAL=5;  S2_FREEZE=15
  S3_EPOCHS=8
fi

NLAYERS=$(echo "$LAYERS" | wc -w)
TAG="${ACT}_h${HIDDEN}_${NLAYERS}layer_${MODE}"
OUT="scratch/$TAG"
mkdir -p "$OUT/stage1" "$OUT/stage2" "$OUT/logs"

echo "### variant act=$ACT hidden=$HIDDEN layers=[$LAYERS] mode=$MODE -> $OUT"
[ "$MODE" = smoke ] && echo "### SMOKE MODE: reduced epochs, will NOT match the paper numbers"

s2ckpts=()
for L in $LAYERS; do
  s1="$OUT/stage1/layer${L}_${ACT}_d${DEG}_h${HIDDEN}_best.pt"
  s2="$OUT/stage2/layer${L}_${ACT}_d${DEG}_h${HIDDEN}_best.pt"

  if [ -f "$s1" ]; then
    echo "### SKIP STAGE1 (exists) $s1"
  else
    echo "### STAGE1 act=$ACT h=$HIDDEN L=$L  $(date)"
    $PY stage1_general_match.py --layer "$L" --hidden "$HIDDEN" --degree $DEG \
        --act "$ACT" --epochs $S1_EPOCHS --recalib-every $S1_RECAL \
        --freeze-epoch $S1_FREEZE --out "$s1" \
        --log "$OUT/logs/stage1_L${L}.log" \
        --results-json "$OUT/logs/stage1_L${L}.json"
  fi

  if [ -f "$s2" ]; then
    echo "### SKIP STAGE2 (exists) $s2"
  else
    echo "### STAGE2 act=$ACT h=$HIDDEN L=$L  $(date)"
    $PY stage2_finetuned_match.py --layer "$L" --hidden "$HIDDEN" --degree $DEG \
        --act "$ACT" --epochs $S2_EPOCHS --recalib-every $S2_RECAL \
        --freeze-epoch $S2_FREEZE --stage1-ckpt "$s1" --teacher-ckpt "$TEACHER" \
        --out "$s2" --log "$OUT/logs/stage2_L${L}.log" \
        --results-json "$OUT/logs/stage2_L${L}.json"
  fi
  s2ckpts+=("$s2")
done

# Stage-3 KD substitution over all LAYERS at once.
# The Bernstein 4-layer recipe gives layer 3 its own (10x) LR and uses alpha=0.7:
# layer 3 receives a much weaker KD gradient than the earlier layers.
extra=()
if [ "$ACT" = "bern" ]; then
  case " $LAYERS " in *" 3 "*) extra=(--alpha 0.7 --bern-lr-l3 1.2e-3 --warmup-frac 0.05);; esac
else
  extra=(--alpha 0.7 --warmup-frac 0.05)
fi

echo "### STAGE3 KD act=$ACT h=$HIDDEN layers=[$LAYERS]  $(date)"
$PY stage3_substitute.py --kd --layers $LAYERS --hidden "$HIDDEN" --degree $DEG \
    --act "$ACT" --epochs $S3_EPOCHS --layer-ckpts "${s2ckpts[@]}" \
    --teacher-ckpt "$TEACHER" --out "$OUT/${ACT}_${NLAYERS}layer_kd_h${HIDDEN}_best.pt" \
    --log "$OUT/logs/stage3.log" --results-json "$OUT/logs/stage3.json" "${extra[@]}"

echo "### DONE act=$ACT h=$HIDDEN layers=[$LAYERS] mode=$MODE  $(date)"
echo "### checkpoint: $OUT/${ACT}_${NLAYERS}layer_kd_h${HIDDEN}_best.pt"
if [ "$MODE" = full ] && [ "$NLAYERS" = 4 ]; then
  echo "### convert to a release with:"
  echo "###   $PY build_release.py --in $OUT/${ACT}_${NLAYERS}layer_kd_h${HIDDEN}_best.pt \\"
  echo "###       --out-dir $OUT --name release_${ACT}_h${HIDDEN}"
fi
