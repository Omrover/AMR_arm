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
