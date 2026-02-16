#!/bin/bash
cd ../

# Comprobar si se han pasado argumentos
if [ $# -eq 0 ]; then
    echo "Error: Debes proporcionar al menos un sufijo de método"
    echo "Uso: $0 METHOD_SUFFIX1 [METHOD_SUFFIX2 ...]"
    echo "Ejemplo: $0 v1 v2 v3"
    exit 1
fi


# Iterar sobre todos los argumentos pasados
for METHOD_SUFFIX in "$@"
do
    METHOD_NAME="fit3D_Base_${METHOD_SUFFIX}"
    RESULTS_FILE="/opt2/data/jzafra/evaluations/${METHOD_NAME}/results.pkl"
    SUMMARY_PATH="/opt2/data/jzafra/metrics/${METHOD_NAME}/"
    
    echo "==============================================="
    echo "Método: ${METHOD_NAME}"
    echo "Resultados en: ${RESULTS_FILE}"
    echo "==============================================="
    
    # Verificar que existe el directorio de predicciones
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "⚠️  ADVERTENCIA: El directorio $RESULTS_FILE no existe. Saltando..."
        echo ""
        continue
    fi
    
    echo "    Generando resumen de resultados..."
    python summary_results.py --results_path "$RESULTS_FILE" \
                                --out_folder "$SUMMARY_PATH"

    echo "✓ Resumen completado para ${METHOD_NAME}"
    echo ""
done

echo "==============================================="
echo "Todas los resumenes han finalizado"
echo "==============================================="