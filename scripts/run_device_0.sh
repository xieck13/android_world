#!/bin/bash

EMULATOR_NAME=AndroidWorldAvd

# Try to find the emulator executable
if [ -f "$HOME/Android/Sdk/emulator/emulator" ]; then
    EMULATOR="$HOME/Android/Sdk/emulator/emulator"
elif [ -f "$ANDROID_HOME/emulator/emulator" ]; then
    EMULATOR="$ANDROID_HOME/emulator/emulator"
else
    echo "Error: Android emulator not found. Please ensure Android SDK is installed and ANDROID_HOME is set."
    exit 1
fi

COMMAND="$EMULATOR -avd $EMULATOR_NAME -no-snapshot -grpc 8554"

while true; do
  echo $COMMAND
  $COMMAND
  
  if [ $? -eq 0 ]; then
    echo "Emulator started successfully"
    break
  else
    echo "Emulator failed to start, retrying in 30 seconds..."
    sleep 30
  fi
done