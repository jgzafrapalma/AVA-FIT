#!/bin/bash
cd ../

# Uso: ./evaluate.sh <dataset> <METHOD_SUFFIX1> [METHOD_SUFFIX2 ...]
# Ejemplo: ./evaluate.sh avafit v1 v2 v3
#          ./evaluate.sh fit3d v1 v2

# Comprobar si se han pasado suficientes argumentos
if [ $# -lt 2 ]; then
    echo "Error: Debes proporcionar el dataset y al menos un sufijo de método"
    echo "Uso: $0 <dataset> METHOD_SUFFIX1 [METHOD_SUFFIX2 ...]"
    echo "Datasets disponibles: avafit, fit3d"
    echo "Ejemplo: $0 avafit v1 v2 v3"
    exit 1
fi

DATASET="$1"
shift  # Quitar el primer argumento (dataset), el resto son sufijos

EVAL_BASE="/opt2/data/jzafra/evaluations"

# Configurar paths según el dataset
case "$DATASET" in
    avafit)
        GT_PATH="/opt2/data/jzafra/gt/avafit"
        METHOD_PREFIX="avafit_Base_"
        ;;
    fit3d)
        GT_PATH="/opt2/data/jzafra/gt/fit3d"
        METHOD_PREFIX="fit3D_Base_"
        ;;
    *)
        echo "Error: Dataset '$DATASET' no reconocido. Usa 'avafit' o 'fit3d'"
        exit 1
        ;;
esac

echo "==============================================="
echo "Dataset: ${DATASET}"
echo "GT Path: ${GT_PATH}"
echo "==============================================="

# Iterar sobre todos los sufijos de método
for METHOD_SUFFIX in "$@"
do
    METHOD_NAME="${METHOD_PREFIX}${METHOD_SUFFIX}"
    PREDS_PATH="/opt2/data/jzafra/predictions/${METHOD_NAME}"
    RESULTS_PATH="${EVAL_BASE}/${METHOD_NAME}/"
    
    echo "==============================================="
    echo "Método: ${METHOD_NAME}"
    echo "Predicciones en: ${PREDS_PATH}"
    echo "==============================================="
    
    # Verificar que existe el directorio de predicciones
    if [ ! -d "$PREDS_PATH" ]; then
        echo "⚠️  ADVERTENCIA: El directorio $PREDS_PATH no existe. Saltando..."
        echo ""
        continue
    fi
    
    echo "    Ejecutando evaluación..."
    python evaluation.py --dataset "$DATASET" \
                         --preds_path "$PREDS_PATH" \
                         --gt_smplx_path "$GT_PATH" \
                         --save_path "$RESULTS_PATH" --overwrite
    
    echo "✓ Evaluación completada para ${METHOD_NAME}"
    echo ""
done

echo "==============================================="
echo "Todas las evaluaciones han finalizado"
echo "==============================================="
