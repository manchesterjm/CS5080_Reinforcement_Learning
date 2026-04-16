# Context Menu Setup for Surface Pro

> **For Claude Code:** Run all registry commands below automatically. These must be run from an **elevated CMD** (admin Command Prompt), not PowerShell. Use `powershell -Command "Start-Process cmd -Verb RunAs"` to get an admin CMD if needed. The Ubuntu distro name may differ on this machine — run `wsl --list` first and substitute the correct name in the Ubuntu entry.
>
> **CRITICAL — `%V` escaping in batch files:** If you run these commands from a `.bat` file, every `%V` must be written as `%%V`, otherwise cmd.exe eats the `%` and the registry ends up with a broken `"V"` instead of `"%V"`. The commands below show the form for a **direct CMD paste**. If you're generating a .bat file, double every `%V` to `%%V`.

## Step 1: Restore Classic Right-Click Menu (Windows 11)

Windows 11 uses a simplified right-click menu. This restores the full classic menu:

```cmd
reg add "HKCU\Software\Classes\CLSID\{86ca1aa0-34aa-4e8b-a509-50c905bae2a2}\InprocServer32" /f /ve
```

## Step 2: Add Custom Context Menu Entries

All commands below require elevated CMD.

### Open Terminal here (Admin)
```cmd
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\cmd_Admin" /ve /d "Open Terminal here (Admin)" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\cmd_Admin" /v HasLUAShield /d "" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\cmd_Admin" /v Icon /d "cmd.exe" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\cmd_Admin\command" /ve /d "powershell -WindowStyle Hidden -Command \"Start-Process wt -ArgumentList '-p \"\"Command Prompt\"\" -d \"\"%%V\"\"' -Verb RunAs\"" /f
```

### Open PowerShell here
```cmd
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\PS_Here" /ve /d "Open PowerShell here" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\PS_Here" /v Icon /d "powershell.exe" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\PS_Here\command" /ve /d "powershell.exe -NoExit -Command Set-Location -LiteralPath '%V'" /f
```

### Open PowerShell here (Admin)
```cmd
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\PS_HereAdmin" /ve /d "Open PowerShell here (Admin)" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\PS_HereAdmin" /v HasLUAShield /d "" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\PS_HereAdmin" /v Icon /d "powershell.exe" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\PS_HereAdmin\command" /ve /d "powershell -Command \"Start-Process powershell -ArgumentList '-NoExit -Command Set-Location ''%V''' -Verb RunAs\"" /f
```

### Open Ubuntu here
**Important:** The distro name must match exactly. Run `wsl --list` first. Replace `Ubuntu-24.04` below with whatever your distro is called.

```cmd
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\Ubuntu" /ve /d "Open Ubuntu here" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\Ubuntu" /v Icon /d "C:\Windows\system32\wsl.exe,0" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\Ubuntu\command" /ve /d "wt.exe -p \"Ubuntu-24.04\" -d \"%V\"" /f
```

### Open WSL here
Opens default WSL distribution. Must be wrapped in `wt.exe` — calling `wsl.exe` directly from a registry shell command spawns a console that closes as soon as wsl exits (no persistent terminal host).

```cmd
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\WSL_Here" /ve /d "Open WSL here" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\WSL_Here" /v Icon /d "C:\Windows\System32\wsl.exe" /f
reg add "HKEY_CLASSES_ROOT\Directory\Background\shell\WSL_Here\command" /ve /d "wt.exe -d \"%V\" wsl.exe" /f
```

## Result

After running all commands, right-clicking in any folder should show:
1. Open in Terminal (cmd) — built-in
2. Open Terminal here (Admin)
3. Open Git GUI here — if Git installed
4. Open Git Bash here — if Git installed
5. Open PowerShell here
6. Open PowerShell here (Admin)
7. Open Ubuntu here
8. Open with Code — if VS Code installed
9. Open WSL here

Changes take effect immediately, no restart needed.
