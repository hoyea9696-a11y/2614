param (
    [string]$TargetFile,
    [string]$ShortcutName,
    [string]$Description
)

# Get the actual desktop path dynamically (handles OneDrive)
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutFile = Join-Path $DesktopPath "$ShortcutName.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutFile)
$Shortcut.TargetPath = $TargetFile
$Shortcut.WorkingDirectory = (Split-Path $TargetFile -Parent)
$Shortcut.Description = $Description
$Shortcut.IconLocation = "imageres.dll, 202" # Vision / Eye Tech Icon
$Shortcut.Save()
