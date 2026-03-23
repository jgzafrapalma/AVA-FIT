#!/bin/bash
cd ../

HOME="/opt2/data/jzafra"

EVALUATIONS_PATH="$HOME/evaluations"
PREDICTIONS_PATH="$HOME/predictions"
METRICS_PATH="$HOME/metrics"

# Parse optional arguments
FPS_FLAG=""
for arg in "$@"; do
    case $arg in
        --fps)
            FPS_FLAG="--fps"
            ;;
    esac
done

echo "==============================================="
echo "Generando métricas..."
echo "==============================================="

python get_metrics.py --results_path "$EVALUATIONS_PATH" \
                      --preds_path "$PREDICTIONS_PATH" \
                      --out_folder "$METRICS_PATH" \
                      $FPS_FLAG

echo "==============================================="
echo "Métricas generadas correctamente"
echo "==============================================="