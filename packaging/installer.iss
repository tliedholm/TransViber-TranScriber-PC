; Inno Setup script for TransViber TransScriber.
; Compile after the PyInstaller build:
;   ISCC.exe packaging\installer.iss
; Output: packaging\dist\TransViber TransScriber Setup.exe

#define AppName "TransViber TransScriber"
#define AppVersion "1.1"
#define AppExe "TransViber TransScriber.exe"
#define DistDir "dist\TransViber TransScriber"

[Setup]
AppId={{29122C2A-6CA7-4E6D-9ED1-306F5F62A38E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=TransViber
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
; The app tree is ~7 GB — solid compression would take forever for little
; gain (model weights barely compress); lzma2/fast is a good middle ground.
Compression=lzma2/fast
SolidCompression=no
; A single Setup.exe caps out at ~4.2 GB on Windows; the app is bigger, so
; the installer ships as Setup.exe + numbered .bin slices (kept together in
; one zip for sharing).
DiskSpanning=yes
DiskSliceSize=2000000000
OutputDir=dist\installer
OutputBaseFilename=TransViber TransScriber Setup
SetupIconFile=..\app\assets\icon.ico
WizardStyle=modern
PrivilegesRequired=admin
; Allows `Setup.exe /CURRENTUSER` to install per-user without admin rights.
PrivilegesRequiredOverridesAllowed=commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
