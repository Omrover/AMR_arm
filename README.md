# AMR_arm
Software for controling 4 DOG amr based on ROS2 an LLC on s32k344 with brushless motors

%% =========================
% PARAMETROS DEL ROBOT
% =========================

l1 =  0.031+0.03885;
l2 = 0.4;
l3 = 0.095;
l4 = 0.245;
l5 = 0.050;

C = sqrt(l3^2 + l4^2);
phi = acos(l4/C);
gamma = pi + phi;



%% VARIABLES ARTICULARES

q1 = deg2rad(0);
q2 = deg2rad(0);
q3 = deg2rad(0);
q4 = deg2rad(0);


%% TABLA DH

DH = [q1       l1   0    pi/2;
      q2       0    l2   0;
      q3-phi   0    -C   0;
      q4+gamma 0    l5   0];

under examples/ you can find an implementation with CAN simple of the movement of the brushless using positions refs.

## `amr_arm_control` (ROS 2 Jazzy)

| Archivo | Qué hace |
|---|---|
| `amr_arm_control/kinematics.py` | FK con la tabla DH de arriba e IK analítica: posición del TCP + pitch de la herramienta (`-pi/2` = vertical hacia abajo). Sin muñeca roll, el yaw lo pone la base |
| `amr_arm_control/trajectory.py` | Punto a punto sincronizado: aceleración por junta para que todas terminen juntas con el perfil trapezoidal del GIM6010-8 |
| `amr_arm_control/arm_server.py` | Acciones `move_arm` (NAMED / JOINT / CARTESIAN), `pick_object` y `place_object` sobre `llc/arm_command` y `llc/set_gripper` del bridge |
| `amr_arm_control/arm_check.py` | Recorrido de verificación: poses con nombre, pick frente al robot y place en la bandeja |
| `config/arm_server.yaml` | Poses con nombre (tentativas, ajustar con el CAD), alturas de pick/place y gripper |

```bash
ros2 launch amr_arm_control arm_server.launch.py                 # robot (bridge corriendo)
ros2 run amr_arm_control arm_check --ros-args -p pick:=false     # solo poses, brazo libre
```

La conversión rad <-> counts y los límites articulares están en
`AMR_LLC/ros2/amr_mqtt_bridge/config/arm.yaml`. El MCU además aplica un límite
de seguridad en counts (`TLALOC_ARM_LIMIT_COUNTS`). Con e-stop, el brazo frena y
mantiene la posición.

El firmware del CAN (protocolo GDZ del ejemplo `examples/main.txt`) vive en
`AMR_LLC/firmware/lib/gim6010`, con simulador y pruebas en host.
