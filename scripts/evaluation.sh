#!/bin/bash
cd ../

HOME="/opt2/data/jzafra"

# Uso: ./evaluate.sh <dataset> <METHOD_SUFFIX1> [METHOD_SUFFIX2 ...] [-- <args extra para evaluation.py>]
# Ejemplo: ./evaluate.sh avafit v1 v2 v3
#          ./evaluate.sh fit3d v1 v2
#          ./evaluate.sh avafit v1 v2 -- --participants P1 P2 --verbose

# Comprobar si se han pasado suficientes argumentos
if [ $# -lt 2 ]; then
    echo "Error: Debes proporcionar el dataset y al menos un sufijo de método"
    echo "Uso: $0 <dataset> METHOD_SUFFIX1 [METHOD_SUFFIX2 ...] [-- <args extra para evaluation.py>]"
    echo ""
    echo "Datasets disponibles: avafit, fit3d"
    echo ""
    echo "Opciones adicionales (se pasan directamente a evaluation.py):"
    echo "  --participants <p1> [p2]    Participantes a procesar"
    echo "  --viewpoints <v1> [v2]      Viewpoints a procesar"
    echo "  --exercises <e1> [e2]       Ejercicios a procesar"
    echo "  --repetitions <r1> [r2]     Repeticiones a procesar (solo avafit)"
    echo "  --verbose                   Mostrar resultados detallados"
    echo "  --overwrite                 Sobreescribir resultados existentes"
    echo ""
    echo "Ejemplo:"
    echo "  $0 avafit v1 v2 v3"
    echo "  $0 avafit v1 v2 -- --participants P1 P2 --verbose"
    echo "  $0 fit3d v1 -- --viewpoints cam1 cam2 --exercises squat"
    exit 1
fi

DATASET="$1"
shift  # Quitar el primer argumento (dataset), el resto son sufijos y posibles args extra

# Separar sufijos de método y argumentos extra (separados por --)
METHOD_SUFFIXES=()
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    if [ "$1" == "--" ]; then
        shift
        EXTRA_ARGS=("$@")
        break
    fi
    METHOD_SUFFIXES+=("$1")
    shift
done

# Comprobar que hay al menos un sufijo de método
if [ ${#METHOD_SUFFIXES[@]} -eq 0 ]; then
    echo "Error: Debes proporcionar al menos un sufijo de método"
    exit 1
fi

EVAL_BASE="${HOME}/evaluations"
GT_PATH="${HOME}/gt"
METHOD_PREFIX="${DATASET}_Base_"

echo "==============================================="
echo "Dataset: ${DATASET}"
echo "GT Path: ${GT_PATH}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Args extra: ${EXTRA_ARGS[*]}"
fi
echo "==============================================="

# Iterar sobre todos los sufijos de método
for METHOD_SUFFIX in "${METHOD_SUFFIXES[@]}"
do
    METHOD_NAME="${METHOD_PREFIX}${METHOD_SUFFIX}"
    PREDS_PATH="${HOME}/predictions/${METHOD_NAME}"
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
                         --save_path "$RESULTS_PATH" \
                         "${EXTRA_ARGS[@]}"
    
    echo "✓ Evaluación completada para ${METHOD_NAME}"
    echo ""
done

echo "==============================================="
echo "Todas las evaluaciones han finalizado"
echo "==============================================="
