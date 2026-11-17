# Lab guide: torque readings and ball balancing

Run top to bottom at the arm. Stop at the first step that fails.
Only the last section (F) moves the arm: keep a partner on the e-stop.

## A. Bring up the station (once per session)
1. Flip the FCI switch UP, turn on the Control PC (blue light) and the user PC. Log in as `student` (password `16-662`).
2. Terminal 1: `ssh -X student@iam-sneezy` (the CONTROL PC; name is on its casing), then `google-chrome`, open `172.16.0.2` (admin / shavethepandas), click unlock. Wait for blue lights.
3. Terminal 2: `roscore`
4. Terminal 3: `cd /home/student/Documents/frankapy && bash ./bash_scripts/start_control_pc.sh -u student -i iam-sneezy` (leave running; restart after any e-stop)
5. Terminal 4: `cd /home/student/16662_RobotAutonomy && bash run_docker.sh`
6. Terminal 5: `cd /home/student/16662_RobotAutonomy && bash terminal_docker.sh` (everything below runs here, inside the container)

## B. Get the code
Outside docker: `cd /home/student/16662_RobotAutonomy/src/devel_packages && git clone https://github.com/RaiPujol/Franka_balancing_ball.git`
(or `git pull` if already cloned). In the container:

    cd src/devel_packages/Franka_balancing_ball/torque_balance
    ls

## C. Only read torques (no motion, no setup)
    python3 check_setup.py     # checks roscore, server and robot, says which step to redo
    python3 torque_watch.py

Hands off for the 2 s zeroing, then press on the paddle. Prints the torque change of joints 1 to 7 in mNm.
A golf ball moved 1 cm changes joints 5 to 7 by about 4.5 mNm; compare with the noise line.
Wrist noise 10 mNm or less: workable. About 20: marginal. Above 40: use a heavier ball.

## D. Mount the paddle, teach the pose
    python3 franka_ball_balance.py grip
    python3 franka_ball_balance.py teach --seconds 30

Hand-guide the paddle level, in free space, then let go before the time runs out.

## E. Richer live readout
    python3 franka_ball_balance.py watch      (Enter = zero again, Ctrl-C = quit)

## F. Learn the arm and balance (moves the arm)
    python3 franka_ball_balance.py tare        # NO ball on the paddle, 60 s
    python3 franka_ball_balance.py calibrate   # ball at centre, then 8 cm out
    python3 franka_ball_balance.py locate      # compare with a ruler
    python3 franka_ball_balance.py balance --seconds 30

## Troubleshooting
- `No module named frankapy`: you are outside the container. Use terminal 5.
- Waits 10 s for the interface: roscore or the control PC server is not running, or the arm is locked / FCI off.
- Chrome errors (viz_main_impl, command_buffer_proxy_impl, message bus, default config): X forwarding noise, ignore. Use `google-chrome --disable-gpu 2>/dev/null &`.
- No lights and no `172.16.0.2`: arm is powered down. FCI switch off, wait 1 min, switch UP, wait 1 to 2 min.
- After an e-stop: unlock in Desk again and restart the control PC server (step A4).
