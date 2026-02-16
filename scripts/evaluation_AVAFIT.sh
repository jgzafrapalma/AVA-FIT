#!/bin/bash
cd ../

# Comprobar si se han pasado argumentos
if [ $# -eq 0 ]; then
    echo "Error: Debes proporcionar al menos un sufijo de método"
    echo "Uso: $0 METHOD_SUFFIX1 [METHOD_SUFFIX2 ...]"
    echo "Ejemplo: $0 v1 v2 v3"
    exit 1
fi

GT_PATH="/opt2/data/jzafra/gt/avafit"

# Iterar sobre todos los argumentos pasados
for METHOD_SUFFIX in "$@"
do
    METHOD_NAME="avafit_Base_${METHOD_SUFFIX}"
    PREDS_PATH="/opt2/data/jzafra/predictions/${METHOD_NAME}"
    RESULTS_PATH="/opt2/data/jzafra/evaluations/${METHOD_NAME}/"
    
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
    python evaluation_AVAFIT.py --preds_path "$PREDS_PATH" \
                                --gt_smplx_path "$GT_PATH" \
                                --save_path "$RESULTS_PATH" --overwrite --participant "001_herong"
    
    echo "✓ Evaluación completada para ${METHOD_NAME}"
    echo ""
done

echo "==============================================="
echo "Todas las evaluaciones han finalizado"
echo "==============================================="