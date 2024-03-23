#!/usr/bin/env bash

SESSION="setup_tmux"
SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)

export robot_hostname=$HOSTNAME

if [ "$SESSIONEXISTS" = "" ]
then
    # Start New Session with our name
    tmux new-session -d -s $SESSION

    tmux rename-window -t 0 'turtlebot-and-perception'

    tmux send-keys 'python ros2_publish_webcam.py' C-m

    tmux split-window -h
    tmux send-keys 'sleep 2' C-m
    tmux send-keys 'sleep 2' C-m

    tmux select-layout even-vertical
fi

# TODO overall 
1. I just want to publish webcam image on ros2
2. Visualise somewhere e.g. ros2, or ros2/1 bridge
3. Then apply contour edge detection etc code and visualise this
4. Then start building control algorithm. 

# if [ "$attach_session" == "y" ]; then
#     # Attach Session, on the Main window
#     tmux attach-session -t $SESSION:0
# else
#     echo "not attaching because attach_session: $attach_session"
# fi

tmux attach-session -t $SESSION:0
