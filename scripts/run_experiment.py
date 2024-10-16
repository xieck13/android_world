import subprocess
import time
import os
import signal

def start_processes():
    # 启动进程A
    command_a = (
        "EMULATOR_NAME=AndroidWorldAvd && "
        "~/Android/Sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554"
    )
    process_a = subprocess.Popen(command_a, shell=True, executable='/bin/bash')
    print("Process A started with PID:", process_a.pid)

    # 等待2分钟后启动进程B
    time.sleep(120)

    command_b = (
        "agent_name=m3a_gpt4o && "
        "python run.py --suite_family=android_world "
        "--agent_name=$agent_name "
        "--output_path ../runs/train_$agent_name "
        "--console_port 5554 "
        "--grpc_port 8554 "
        "--checkpoint_dir ../checkpoint/train_$agent_name "
        "--tasks SimpleCalendarDeleteEventsOnRelativeDay,SimpleCalendarEventOnDateAtTime,SimpleCalendarEventsOnDate,SimpleCalendarNextMeetingWithPerson,SimpleSmsReplyMostRecent,SimpleSmsResend,SimpleSmsSend,SimpleSmsSendReceivedAddress,SportsTrackerActivitiesOnDate,SportsTrackerTotalDistanceForCategoryOverInterval,SportsTrackerTotalDurationForCategoryThisWeek,TasksCompletedTasksForDate,TasksDueNextWeek,TasksHighPriorityTasks,TasksHighPriorityTasksDueOnDate,TurnOffWifiAndTurnOnBluetooth,VlcCreatePlaylist,VlcCreateTwoPlaylists"
    )
    process_b = subprocess.Popen(command_b, shell=True, executable='/bin/bash')
    print("Process B started with PID:", process_b.pid)

    return process_a, process_b

def kill_process(process):
    if process and process.poll() is None:
        os.kill(process.pid, signal.SIGTERM)
        print(f"Process with PID {process.pid} killed.")

def monitor_processes():
    process_a, process_b = start_processes()

    while True:
        # 检查进程A和进程B是否存活
        if process_a.poll() is not None:
            print("Process A has terminated.")
            kill_process(process_b)
            process_a, process_b = start_processes()

        if process_b.poll() is not None:
            print("Process B has terminated.")
            kill_process(process_a)
            process_a, process_b = start_processes()

        # 每隔10秒检查一次
        time.sleep(10)

if __name__ == "__main__":
    monitor_processes()
