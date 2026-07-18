# mrhack - verified control glue (mock/sim tested)

Drop-in control spine for the K1 x projector-AR nav demo. Every core is verified in
mock/sim with NO hardware; it binds to the real robot by swapping the mock client for
the SDK and the sim pose for Vive/odometry.

- contracts.py  - shared dataclasses + JSON wire (RobotPose, Goal, Trajectory, Setpoint, VelCmd, CalibConfig)
- bus.py        - ZeroMQ XPUB/XSUB proxy + pub/sub + drain_latest  (selftest PASSES)
- controller/pure_pursuit.py (M5) - lookahead IS the projected circle  (3/3 convergence)
- actuator/state_machine.py  (M7) - K1Gate DAMP->PREP->WALK, clamps, guaranteed safe stop
- actuator/velocity_id.py         - FIRST runnable on the robot: open-loop velocity ID (--mock PASSES)
- planner/plan.py            (M4) - straight-line path
- sim/kinematic_sim.py            - full loop plan->pursuit->sim  (4/4, incl. latency + velocity mistracking)
- starters/calibrate.py     (M1) - camera->floor + projector->floor homographies (--selftest 0.0mm)
- starters/vive_rerun.py    (M2) - Vive -> pose -> Rerun (needs hardware)
- starters/oakd_detect.py  (M3a) - OAK-D object detection (needs hardware)

Reproduce (no hardware):
    python -m mrhack.controller.pure_pursuit
    python -m mrhack.sim.kinematic_sim
    python -m mrhack.actuator.velocity_id --mock --settle 0.3 --cmd 0.5 --matrix "0.2,0.0;0.0,0.3"
    uv run --with pyzmq python -m mrhack.bus selftest
    uv run --with opencv-python --with numpy python starters/calibrate.py --selftest

On the real robot: bind K1Gate.make_client() to the ROS/boosteros surface; feed pose from Vive/odometry.
