#!/bin/bash
cd ../

# Comprobar si se han pasado suficientes argumentos (Mínimo métrica + 1 sufijo)
if [ $# -lt 2 ]; then
    echo "Error: Debes proporcionar la MÉTRICA y al menos un sufijo de método."
    echo "Uso: $0 METRIC METHOD_SUFFIX1 [METHOD_SUFFIX2 ...]"
    echo "Ejemplo: $0 MPJPE v1 v2 v3"
    exit 1
fi

# 1. Asignar el primer argumento a la variable METRIC
METRIC=$1

# 2. Desplazar los argumentos. $2 pasa a ser $1, $3 a $2, etc.
# Esto saca la métrica de la lista "$@" para que el bucle solo itere los sufijos.
shift

# Iterar sobre todos los argumentos restantes (los sufijos)
for METHOD_SUFFIX in "$@"
do
    METHOD_NAME="avafit_Base_${METHOD_SUFFIX}"
    RESULTS_FILE="/opt2/data/jzafra/evaluations/${METHOD_NAME}/results.pkl"
    SAVE_PATH="/opt2/data/jzafra/errors/${METHOD_NAME}/${METRIC}"
    
    echo "==============================================="
    echo "Método: ${METHOD_NAME}"
    echo "Métrica: ${METRIC}"
    echo "Evaluaciones en: ${RESULTS_FILE}"
    echo "==============================================="
    
    # Verificar que existe el FICHERO de predicciones (-f para ficheros, -d para directorios)
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "⚠️  ADVERTENCIA: El fichero de evaluaciones $RESULTS_FILE no existe. Saltando..."
        echo ""
        continue
    fi
    
    # Crear el directorio de salida si no existe (opcional pero recomendado)
    if [ ! -d "$SAVE_PATH" ]; then
        echo "    Creando directorio de salida: $SAVE_PATH"
        mkdir -p "$SAVE_PATH"
    fi
    
    echo "    Ejecutando obtención de errores..."

    python get_errors_avafit.py --evaluation_file "$RESULTS_FILE" \
                            --metrics "$METRIC" \
                            --output_path "$SAVE_PATH"

    
    echo "✓ Top Errors obtenidos para ${METHOD_NAME}"
    echo ""
done

echo "==============================================="
echo "Las obtenciones de errores han finalizado"
echo "==============================================="