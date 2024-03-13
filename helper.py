import sys
import os
import time

def user_confirm():
    start_input = time.time()
    while True:
        confirm = input('Continue? (Y/n)')
        if (confirm.lower() == 'y') | (confirm.lower()=='yes') | (confirm==''):
            break
        elif (confirm.lower()=='n') | (confirm.lower()=='no'):
            print("Stopping")
            sys.exit(2)
        else:
            print("Please enter yes(Y) or no(N)")
    user_time = time.time() - start_input
    return user_time

def which_trajs(PD):
    if 'trajs' not in PD.__dict__.keys():
        print("List of trajectories not provided. All will be parsed...")
        PD.trajs = [t.name for t in os.scandir(PD.traj_dir) if t.name.isnumeric()]
    print("The following directories will be parsed as trajectories:")
    for traj in PD.trajs:
        print(PD.traj_dir+traj+'\n')
    user_time = user_confirm()
    return user_time