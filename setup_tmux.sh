#!/usr/bin/env bash

SESSION="setup_tmux"
SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)

export robot_hostname=$HOSTNAME

if [ "$SESSIONEXISTS" = "" ]
then
    # Start New Session with our name
    tmux new-session -d -s $SESSION

    tmux rename-window -t 0 'turtlebot-and-perception'

    tmux send-keys 'python3 ros2_publish_webcam.py' C-m

    tmux split-window -h
    tmux send-keys 'sleep 2' C-m
    tmux send-keys 'python3 ros2_subscribe_webcam_and_find_lines.py' C-m

    tmux select-layout even-vertical
fi

#TODO running tmux ruins my ros setup and makes the code not work somehow. Need to get to the bottom of this. Ask gpt

# if [ "$attach_session" == "y" ]; then
#     # Attach Session, on the Main window
#     tmux attach-session -t $SESSION:0
# else
#     echo "not attaching because attach_session: $attach_session"
# fi

tmux attach-session -t $SESSION:0

# run within docker
# docker run --runtime nvidia -it --rm --network host --volume /tmp/argus_socket:/tmp/argus_socket --volume /etc/enctune.conf:/etc/enctune.conf --volume /etc/nv_tegra_release:/etc/nv_tegra_release --volume /tmp/nv_jetson_model:/tmp/nv_jetson_model --volume /var/run/dbus:/var/run/dbus --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket --volume /var/run/docker.sock:/var/run/docker.sock --volume /home/general/all_projects/jetson-containers/packages/llm/local_llm:/opt/local_llm/local_llm --volume /home/general/all_projects/jetson-containers/data:/data --device /dev/snd --device /dev/bus/usb --device /dev/video0 --volume /home/general/share_with_docker/:/share_with_docker my_latest_galactic