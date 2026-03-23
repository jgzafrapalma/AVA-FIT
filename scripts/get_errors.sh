#!/bin/bash
cd ../

HOME="/opt2/data/jzafra"

if [ $# -lt 2 ]; then
    echo "Error: Debes proporcionar la MÉTRICA, N (top errors)"
    echo "Uso: $0 METRIC N"
    echo "Ejemplo: $0 MPJPE 10"
    exit 1
fi

METRIC=$1
shift

N=$1
shift

EVALUATION_PATH="${HOME}/evaluations/"
SAVE_PATH="${HOME}/errors/"
    
echo "==============================================="
echo "Métrica: ${METRIC}"
echo "Top N: ${N}"
echo "==============================================="
    
# Verificar que existe el directorio de evaluaciones
if [ ! -d "$EVALUATION_PATH" ]; then
    echo "⚠️  ADVERTENCIA: El directorio de evaluaciones $EVALUATION_PATH no existe. Saltando..."
    echo ""
    continue
fi

# Crear el directorio de salida si no existe (opcional pero recomendado)
if [ ! -d "$SAVE_PATH" ]; then
    echo "    Creando directorio de salida: $SAVE_PATH"
    mkdir -p "$SAVE_PATH"
fi

echo "    Ejecutando obtención de errores..."

python get_errors.py --evaluation_path "$EVALUATION_PATH" \
                        --metric "$METRIC" \
                        --output_path "$SAVE_PATH" \
                        --n "$N"

echo "==============================================="
echo "Las obtenciones de errores han finalizado"
echo "==============================================="