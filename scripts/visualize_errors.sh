#!/bin/bash
cd ../

# Comprobar si se han pasado suficientes argumentos (Métrica + Tipo Error + Al menos 1 sufijo)
if [ $# -lt 3 ]; then
    echo "Error: Faltan argumentos."
    echo "Uso: $0 METRIC ERROR_TYPE METHOD_SUFFIX1 [METHOD_SUFFIX2 ...]"
    echo "Ejemplo: $0 MPJPE top_error v1 v2 v3"
    exit 1
fi

# 1. Asignar los primeros argumentos a variables
METRIC=$1
ERROR_TYPE=$2

# 2. Desplazar los argumentos 2 posiciones. 
# $3 pasa a ser $1 (el primer sufijo).
shift 2

GT_PATH="/opt2/data/jzafra/gt/fit3d"
DATASET_PATH="/opt2/data/jzafra/datasets/fit3d"

# Iterar sobre todos los argumentos restantes (los sufijos)
for METHOD_SUFFIX in "$@"
do
    METHOD_NAME="fit3D_Base_${METHOD_SUFFIX}"
    
    # Aquí ahora usamos la variable ${METRIC} que hemos leído al principio
    ERRORS_FILE="/opt2/data/jzafra/errors/${METHOD_NAME}/${METRIC}/${ERROR_TYPE}.pkl"
    
    # Es recomendable incluir la métrica en el path de salida para no mezclar visualizaciones
    SAVE_PATH="/opt2/data/jzafra/visualization/${METHOD_NAME}/${METRIC}/${ERROR_TYPE}"
    PREDS_PATH="/opt2/data/jzafra/predictions/${METHOD_NAME}"
    
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