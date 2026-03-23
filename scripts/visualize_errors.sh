#!/bin/bash
cd ../

HOME="/opt2/data/jzafra"

# Comprobar si se han pasado suficientes argumentos (Métrica + Tipo Error + Al menos 1 sufijo)
if [ $# -lt 4 ]; then
    echo "Error: Faltan argumentos."
    echo "Uso: $0 DATASET METRIC ERROR_TYPE METHOD_SUFFIX1 [METHOD_SUFFIX2 ...]"
    echo "Ejemplo: $0 fit3d MPJPE top_error v1 v2 v3"
    exit 1
fi

DATASET=$1
METRIC=$2
ERROR_TYPE=$3

shift 3

GT_PATH="${HOME}/gt/${DATASET}"
DATASET_PATH="${HOME}/datasets/${DATASET}"

# Iterar sobre todos los argumentos restantes (los sufijos)
for METHOD_SUFFIX in "$@"
do
    METHOD_NAME="${DATASET}_Base_${METHOD_SUFFIX}"
    
    # Aquí ahora usamos la variable ${METRIC} que hemos leído al principio
    ERRORS_FILE="${HOME}/errors/${METHOD_NAME}_${METRIC}/${ERROR_TYPE}_error.json"
    
    # Es recomendable incluir la métrica en el path de salida para no mezclar visualizaciones
    SAVE_PATH="${HOME}/visualization/${METHOD_NAME}_${METRIC}/${ERROR_TYPE}"
    PREDS_PATH="${HOME}/predictions/${METHOD_NAME}"
    
    echo "==============================================="
    echo "Método: ${METHOD_NAME}"
    echo "Métrica: ${METRIC} | Tipo Error: ${ERROR_TYPE}"
    echo "Leyendo errores de: ${ERRORS_FILE}"
    echo "==============================================="
    
    # Verificar que existe el FICHERO de errores
    if [ ! -f "$ERRORS_FILE" ]; then
        echo "⚠️  ADVERTENCIA: El fichero de errores $ERRORS_FILE no existe. Saltando..."
        echo ""
        continue
    fi
    
    # Crear carpeta de salida si no existe
    if [ ! -d "$SAVE_PATH" ]; then
        mkdir -p "$SAVE_PATH"
    fi
    
    echo "    Generando visualización de errores..."

    python visualize_errors.py --errors_path "$ERRORS_FILE" \
                                --out_folder "$SAVE_PATH" \
                                --pred_path "$PREDS_PATH" \
                                --gt_path "$GT_PATH" \
                                --dataset_path "$DATASET_PATH"

    
    echo "✓ Visualizaciones generadas en: $SAVE_PATH"
    echo ""
done

echo "==============================================="
echo "Las visualizaciones de errores han finalizado"
echo "==============================================="