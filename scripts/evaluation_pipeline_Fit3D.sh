#!/bin/bash

METHOD_SUFFIX=$1
PARTICIPANT=$2
METRIC=$3
ACTION=$4        # 0: eval, 1: get_errors, 2: summary, 3: visualize, 4: all
ERROR_TYPE=${5:-top}  # Tipo de error a visualizar: top, mean, median (por defecto: top)

if [ -z "$METHOD_SUFFIX" ] || [ -z "$PARTICIPANT" ] || [ -z "$METRIC" ] || [ -z "$ACTION" ]; then
  echo "Uso: $0 <METHOD_SUFFIX> <PARTICIPANT> <METRIC> <ACTION> [ERROR_TYPE]"
  echo "Ejemplo: $0 PromptHMR s11 PA-MPVPE 4 top"
  echo "ACTION: 0 = evaluar, 1 = errores, 2 = resumen, 3 = visualizar errores, 4 = todo"
  echo "ERROR_TYPE: top (default), mean, median"
  exit 1
fi

METHOD_NAME="fit3D_Base_${METHOD_SUFFIX}"
PREDS_PATH="/opt2/data/jzafra/predictions/${METHOD_NAME}"
GT_PATH="/opt2/data/jzafra/tmp/GT_FIT3D_Camera"
DATASET_PATH="/opt2/data/jzafra/datasets/fit3d"

# Ruta de resultados por participante y métrica
SAVE_PATH="/opt2/data/jzafra/tmp/${METHOD_NAME}_results/${PARTICIPANT}/${METRIC}"

# Ruta del resumen (independiente de la métrica)
SUMMARY_PATH="/opt2/data/jzafra/tmp/${METHOD_NAME}_results/${PARTICIPANT}/summary"

# Crear directorios si no existen
mkdir -p "$SAVE_PATH"
mkdir -p "$SUMMARY_PATH"

cd ../

if [ "$ACTION" -eq 0 ] || [ "$ACTION" -eq 4 ]; then
  echo "Ejecutando evaluación..."
  python evaluation_Fit3D.py --preds_path "$PREDS_PATH" \
                             --gt_smplx_path "$GT_PATH" \
                             --save_path "$SAVE_PATH" \
                             --participants "$PARTICIPANT"
fi

if [ "$ACTION" -eq 1 ] || [ "$ACTION" -eq 4 ]; then
  echo "Obteniendo errores..."
  python get_errors.py --evaluation_file "$SAVE_PATH/results.pkl" \
                       --metrics "$METRIC" --output_path "$SAVE_PATH"
fi

if [ "$ACTION" -eq 2 ] || [ "$ACTION" -eq 4 ]; then
  echo "Generando resumen de resultados..."
  python summary_results.py --results_path "$SAVE_PATH/results.pkl" \
                            --out_folder "$SUMMARY_PATH"
fi

if [ "$ACTION" -eq 3 ] || [ "$ACTION" -eq 4 ]; then
  ERROR_FILE="${ERROR_TYPE}_error.pkl"
  echo "Visualizando errores: $ERROR_FILE"
  python visualize_errors.py --errors_path "$SAVE_PATH/$ERROR_FILE" \
                             --out_folder "$SAVE_PATH/${ERROR_TYPE}_error_visualize" \
                             --pred_path "$PREDS_PATH" \
                             --gt_path "$GT_PATH" \
                             --dataset_path "$DATASET_PATH"
fi

echo "Proceso completado."
