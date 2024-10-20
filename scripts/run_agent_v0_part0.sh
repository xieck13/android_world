easy_task_part0=AudioRecorderRecordAudio,AudioRecorderRecordAudioWithFileName,BrowserDraw,BrowserMaze,CameraTakePhoto,ClockStopWatchPausedVerify,ClockStopWatchRunning,ClockTimerEntry,ContactsAddContact,ContactsNewContactDraft,ExpenseAddSingle,ExpenseDeleteMultiple,ExpenseDeleteSingle,MarkorCreateFolder,MarkorDeleteAllNotes,MarkorDeleteNewestNote,MarkorDeleteNote,MarkorEditNote,NotesIsTodo,NotesMeetingAttendeeCount
easy_task_part1=NotesRecipeIngredientCount,OpenAppTaskEval,RecipeAddSingleRecipe,RecipeDeleteDuplicateRecipes,RecipeDeleteMultipleRecipes,RecipeDeleteSingleRecipe,RecipeDeleteSingleWithRecipeWithNoise,RetroPlayingQueue,SimpleCalendarAddOneEventRelativeDay,SimpleCalendarAddOneEventTomorrow,SimpleCalendarAddRepeatingEvent,SimpleCalendarAnyEventsOnDate,SimpleCalendarDeleteEvents,SimpleCalendarDeleteOneEvent,SimpleCalendarEventsInNextWeek,SimpleCalendarEventsInTimeRange,SimpleCalendarFirstEventAfterStartTime,SimpleCalendarLocationOfEvent,SimpleCalendarNextEvent,SimpleDrawProCreateDrawing
easy_task_part2=SimpleSmsReply,SimpleSmsSendClipboardContent,SportsTrackerActivitiesCountForWeek,SportsTrackerActivityDuration,SportsTrackerLongestDistanceActivity,SystemBluetoothTurnOff,SystemBluetoothTurnOffVerify,SystemBluetoothTurnOn,SystemBluetoothTurnOnVerify,SystemBrightnessMax,SystemBrightnessMaxVerify,SystemBrightnessMin,SystemBrightnessMinVerify,SystemCopyToClipboard,SystemWifiTurnOff,SystemWifiTurnOffVerify,SystemWifiTurnOn,SystemWifiTurnOnVerify,TasksDueOnDate,TasksIncompleteTasksOnDate,TurnOnWifiAndOpenApp


# python run.py   --suite_family=android_world \
#   --agent_name=m3a4_llava_ft \
#   --output_path ../runs/m3a4_llava_ft \
#   --console_port 5554 \
#   --grpc_port 8554

# python run.py   --suite_family=android_world \
#   --agent_name=m3a4_llava_ft \
#   --output_path ../runs/m3a4_llava_ft \
#   --console_port 5554 \
#   --grpc_port 8554 \
#   --tasks ExpenseDeleteDuplicates,ExpenseDeleteDuplicates2,ExpenseDeleteMultiple,ExpenseDeleteMultiple2,ExpenseDeleteSingle,FilesDeleteFile,FilesMoveFile,MarkorAddNoteHeader,MarkorChangeNoteContent,MarkorCreateFolder,MarkorCreateNote,MarkorCreateNoteAndSms,MarkorCreateNoteFromClipboard,MarkorDeleteAllNotes,MarkorDeleteNewestNote,MarkorDeleteNote,MarkorEditNote,MarkorMergeNotes,MarkorMoveNote,MarkorTranscribeReceipt,MarkorTranscribeVideo,NotesIsTodo,NotesMeetingAttendeeCount,NotesRecipeIngredientCount,NotesTodoItemCount,OpenAppTaskEval,OsmAndFavorite,OsmAndMarker,OsmAndTrack,RecipeAddMultipleRecipes,RecipeAddMultipleRecipesFromImage,RecipeAddMultipleRecipesFromMarkor,RecipeAddMultipleRecipesFromMarkor2,RecipeAddSingleRecipe,RecipeDeleteDuplicateRecipes,RecipeDeleteDuplicateRecipes2,RecipeDeleteDuplicateRecipes3,RecipeDeleteMultipleRecipes,RecipeDeleteMultipleRecipesWithConstraint,RecipeDeleteMultipleRecipesWithNoise,RecipeDeleteSingleRecipe,RecipeDeleteSingleWithRecipeWithNoise,RetroCreatePlaylist,RetroPlayingQueue,RetroPlaylistDuration,RetroSavePlaylist,SaveCopyOfReceiptTaskEval,SimpleCalendarAddOneEvent,SimpleCalendarAddOneEventInTwoWeeks,SimpleCalendarAddOneEventRelativeDay,SimpleCalendarAddOneEventTomorrow,SimpleCalendarAddRepeatingEvent,SimpleCalendarAnyEventsOnDate,SimpleCalendarDeleteEvents,SimpleCalendarDeleteEventsOnRelativeDay,SimpleCalendarDeleteOneEvent,SimpleCalendarEventOnDateAtTime,SimpleCalendarEventsInNextWeek,SimpleCalendarEventsInTimeRange,SimpleCalendarEventsOnDate,SimpleCalendarFirstEventAfterStartTime,SimpleCalendarLocationOfEvent,SimpleCalendarNextEvent,SimpleCalendarNextMeetingWithPerson,SimpleDrawProCreateDrawing,SimpleSmsReply,SimpleSmsReplyMostRecent,SimpleSmsResend,SimpleSmsSend,SimpleSmsSendClipboardContent,SimpleSmsSendReceivedAddress,SportsTrackerActivitiesCountForWeek,SportsTrackerActivitiesOnDate,SportsTrackerActivityDuration,SportsTrackerLongestDistanceActivity,SportsTrackerTotalDistanceForCategoryOverInterval,SportsTrackerTotalDurationForCategoryThisWeek,SystemBluetoothTurnOff,SystemBluetoothTurnOffVerify,SystemBluetoothTurnOn,SystemBluetoothTurnOnVerify,SystemBrightnessMax,SystemBrightnessMaxVerify,SystemBrightnessMin,SystemBrightnessMinVerify,SystemCopyToClipboard,SystemWifiTurnOff,SystemWifiTurnOffVerify,SystemWifiTurnOn,SystemWifiTurnOnVerify,TasksCompletedTasksForDate,TasksDueNextWeek,TasksDueOnDate,TasksHighPriorityTasks,TasksHighPriorityTasksDueOnDate,TasksIncompleteTasksOnDate,TurnOffWifiAndTurnOnBluetooth,TurnOnWifiAndOpenApp,VlcCreatePlaylist,VlcCreateTwoPlaylists


# python run.py   --suite_family=miniwob \
#   --agent_name=m3a4_llava_ft \
#   --output_path ../runs/m3a4_llava_ft \
#   --console_port 5554 \
#   --grpc_port 8554 \


# python run.py   --suite_family=android_world \
#   --agent_name=m3a_qwen_72b \
#   --output_path ../runs/m3a_qwen_72b \
#   --console_port 5554 \
#   --grpc_port 8554 \
#   --tasks OsmAndMarker,OsmAndTrack,RetroCreatePlaylist,RetroPlayingQueue,RetroPlaylistDuration,SimpleSmsSend,SystemCopyToClipboard,SystemWifiTurnOff,SystemWifiTurnOffVerify,SystemWifiTurnOn,SystemWifiTurnOnVerify,TasksCompletedTasksForDate


  # python run.py   --suite_family=android_world \
  # --agent_name=m3a4_qwen_72b \
  # --output_path ../runs/m3a4_qwen_72b \
  # --console_port 5554 \
  # --grpc_port 8554 \

# agent_name=m3a4_llava_ft
# python run.py   --suite_family=android_world \
#   --agent_name=$agent_name \
#   --output_path ../runs/$agent_name \
#   --console_port 5554 \
#   --grpc_port 8554 \
#   --tasks SystemBrightnessMaxVerify,SystemBluetoothTurnOffVerify,SystemBrightnessMinVerify


# agent_name=m3a4_qwen_72b
# python run.py   --suite_family=android_world \
#   --agent_name=$agent_name \
#   --output_path ../runs/$agent_name \
#   --console_port 5554 \
#   --grpc_port 8554 \
#   --tasks SystemBrightnessMaxVerify,SystemBluetoothTurnOffVerify,SystemBrightnessMinVerify


# agent_name=m3a_gpt4o
# python run.py   --suite_family=android_world \
#   --agent_name=$agent_name \
#   --output_path ../runs/train_$agent_name \
#   --console_port 5554 \
#   --grpc_port 8554 \
#   --tasks 


agent_name=m3a4_llava_ft
python run.py   --suite_family=android_world \
  --agent_name=$agent_name \
  --output_path ../runs/$agent_name \
  --checkpoint_dir ../checkpoint/$agent_name \
  --console_port 5554 \
  --grpc_port 8554 \


agent_name=m3a6_llava_ft
python run.py   --suite_family=android_world \
  --agent_name=$agent_name \
  --output_path ../runs/$agent_name \
  --checkpoint_dir ../checkpoint/$agent_name \
  --console_port 5554 \
  --grpc_port 8554 \
  