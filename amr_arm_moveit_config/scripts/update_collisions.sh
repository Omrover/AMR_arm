#!/bin/bash
# Regenera la matriz de colisiones deshabilitadas del SRDF desde la URDF actual
# (muestreo aleatorio de poses, sin GUI). Correr desde la raíz del paquete
# fuente tras cambiar la geometría (p. ej. el export de SolidWorks):
#   bash scripts/update_collisions.sh [trials]
set -e
cd "$(dirname "$0")/.."
URDF=/tmp/omrover_mm_collisions.urdf
xacro "$(ros2 pkg prefix amr_description)/share/amr_description/urdf/amr.urdf.xacro" > "$URDF"
ros2 run moveit_setup_assistant collisions_updater \
  --urdf "$URDF" --srdf config/omrover_mm.srdf --output config/omrover_mm.srdf \
  --trials "${1:-20000}" --always
# Sin --default a propósito: deshabilitaría los pares que chocan en q = 0 (con
# la geometría provisional, la cámara contra arm_link_2) y MoveIt dejaría de
# revisarlos en todas las poses.
echo "SRDF actualizado: revisar el diff de config/omrover_mm.srdf"
