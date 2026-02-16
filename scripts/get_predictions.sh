#!/bin/bash
# Script unificado para obtener predicciones de cualquier método en cualquier dataset.
# Uso: ./get_predictions.sh <dataset> <method> [opciones adicionales...]
# Ejemplo: ./get_predictions.sh avafit multihmr --device 0 --save_mesh
#          ./get_predictions.sh fit3d SAM3D_BODY --device 1 --participants s03 s04

cd ../

# Comprobar si se han pasado suficientes argumentos
if [ $# -lt 2 ]; then
    echo "Error: Debes proporcionar el dataset y el método"
    echo "Uso: $0 <dataset> <method> [opciones adicionales...]"
    echo ""
    echo "Datasets disponibles: avafit, fit3d"
    echo "Métodos disponibles: multihmr, 4D-humans, PromptHMR, SAM3D_BODY"
    echo ""
    echo "Opciones adicionales (se pasan directamente al script de Python):"
    echo "  --device <int>              GPU a utilizar (default: 0)"
    echo "  --cpu_cores <str>           Cores de CPU para taskset (default: 20-39,60-79)"
    echo "  --participants <p1> [p2]    Participantes a procesar"
    echo "  --camera_ids <c1> [c2]      Cámaras a procesar"
    echo "  --exercises <e1> [e2]       Ejercicios a procesar"
    echo "  --repetitions <r1> [r2]     Repeticiones a procesar (solo AVAFIT)"
    echo "  --checkpoint <path>         Path al checkpoint"
    echo "  --model_name <name>         Nombre del modelo"
    echo "  --batch_size <int>          Tamaño del batch"
    echo "  --save_mesh                 Guardar ficheros .obj con las mallas"
    echo "  --render                    Guardar renders"
    echo "  --extra_views               Renderizar vistas adicionales"
    echo ""
    echo "Ejemplo:"
    echo "  $0 avafit multihmr --device 0 --cpu_cores \"0-11,24-35\" --save_mesh"
    echo "  $0 fit3d 4D-humans --device 1 --participants s08 --save_mesh --batch_size 8"
    exit 1
fi

DATASET="$1"
METHOD="$2"
shift 2  # Quitar dataset y method, el resto son opciones adicionales

# Configurar paths según el dataset
case "$DATASET" in
    avafit)
        DATASET_PATH="/opt2/data/jzafra/datasets/avafit/"
        OUTPUT_PATH="/opt4/data/jzafra/predictions/avafit_Base_${METHOD}/"
        PYTHON_SCRIPT="predictions_AVAFIT.py"
        ;;
    fit3d)
        DATASET_PATH="/opt2/data/jzafra/datasets/fit3d/"
        OUTPUT_PATH="/opt2/data/jzafra/predictions/fit3D_Base_${METHOD}/"
        PYTHON_SCRIPT="predictions_Fit3D.py"
        ;;
    *)
        echo "Error: Dataset '$DATASET' no reconocido. Usa 'avafit' o 'fit3d'"
        exit 1
        ;;
esac

echo "==============================================="
echo "Dataset:      ${DATASET}"
echo "Método:       ${METHOD}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output path:  ${OUTPUT_PATH}"
echo "Script:       ${PYTHON_SCRIPT}"
echo "Args extra:   $@"
echo "==============================================="

python ${PYTHON_SCRIPT} --dataset_path "${DATASET_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --method "${METHOD}" "$@"
