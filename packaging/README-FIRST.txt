TransViber TransScriber
=======================

An app that turns audio recordings into a written transcript — including
multi-speaker recordings where each person is a separate audio file or
channel (like Discord/Craig recordings). Everything runs on your own
computer; nothing is uploaded anywhere.

GETTING STARTED
---------------
There are two versions of this app. YOU ONLY NEED ONE OF THEM — they are
the same program, so pick whichever suits you and ignore the other. Do not
install both.

  INSTALLER ("...Setup.zip")
  Unzip it, keeping every file in one folder (there's a Setup.exe and
  several numbered .bin files that belong together). Run "TransViber
  TransScriber Setup.exe" and follow the steps. You get a Start-menu
  shortcut, optionally a desktop shortcut, and it appears in the usual
  Windows list of installed programs if you ever want to remove it.

  PORTABLE ("...Portable.zip")
  Unzip the folder anywhere you like — including a USB or external drive —
  and double-click "TransViber TransScriber.exe" inside it. Nothing is
  installed and no shortcuts are made; to get rid of it, delete the folder.
  Useful if you can't install software on the PC.

Either way the app needs about 7 GB of free space once unpacked.

IF WINDOWS SHOWS A BLUE WARNING
-------------------------------
The first time you run it, Windows may show "Windows protected your PC".
That's just because this app isn't from a big registered company. Click
"More info", then "Run anyway". You only have to do this once.

USING IT
--------
1. Drag your audio files into the window (one file per speaker, or one
   combined file — right-click a track to split channels into speakers).
2. Type each speaker's name in the Speaker column.
3. Optional: type unusual names and words (character names, places, jargon)
   into the "Vocabulary hints" box on the left, one per line. The app spells
   them correctly instead of guessing.
4. Click Transcribe. The merged conversation appears below.
5. Export… saves it as text, Markdown, subtitles, and more.

The app comes with the "distil-large-v3.5-ct2" model preloaded, so it works
immediately — no download needed (English only). Picking any other model
from the dropdown downloads it automatically the first time you use it
(they range from ~0.5 to 3 GB; they're saved so it only happens once). The
star button next to the dropdown makes a model the one that's picked every
time you open the app.

There's a dark-mode checkbox at the bottom right.

THE NOTES TAB
-------------
The second tab is a notepad that saves itself automatically, with the
transcript shown alongside for reference. "AI Notes" hands the transcript to
an AI running on your own PC, which writes up a summary, the key points,
notable quotes, follow-ups, and a rundown per person — added into the
notepad while you watch. You can add your own instructions ("focus on plot
events") in the box on the left. "Export Notes…" saves the notepad to a
file. The AI model choice is a dropdown; bigger models are slower but
better, and each one downloads the first time it's used.

GRAPHICS CARD
-------------
If your PC has an NVIDIA graphics card, transcription uses it and is
much faster. Otherwise everything runs on the CPU — same results, just
slower. The "AI Notes" feature (Notes tab) needs an NVIDIA card.

TROUBLESHOOTING
---------------
If the app won't start or crashes, run "TransViber Debug.exe" (in the app
folder) instead — it opens with a black console window that shows what went
wrong. Send a screenshot of that to whoever gave you the app.
