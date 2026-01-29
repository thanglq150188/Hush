@echo off
:: Sync changed files from D:\Hush -> D:\callbot\hush
:: Works both before and after commit

cd /d D:\Hush
setlocal enabledelayedexpansion

set count=0

:: 1. Uncommitted changes (modified + staged)
for /f "delims=" %%f in ('git diff --name-only HEAD 2^>nul') do (
    if not "%%f"=="sync_to_internal.bat" (
        xcopy /Y /Q "D:\Hush\%%f" "D:\callbot\hush\%%f*" >nul 2>&1
        set /a count+=1
    )
)

:: 2. Staged files
for /f "delims=" %%f in ('git diff --name-only --cached 2^>nul') do (
    if not "%%f"=="sync_to_internal.bat" (
        xcopy /Y /Q "D:\Hush\%%f" "D:\callbot\hush\%%f*" >nul 2>&1
        set /a count+=1
    )
)

:: 3. Untracked new files
for /f "delims=" %%f in ('git ls-files --others --exclude-standard 2^>nul') do (
    if not "%%f"=="sync_to_internal.bat" (
        xcopy /Y /Q "D:\Hush\%%f" "D:\callbot\hush\%%f*" >nul 2>&1
        set /a count+=1
    )
)

:: 4. Last commit changes (for running after commit)
if !count!==0 (
    for /f "delims=" %%f in ('git diff --name-only HEAD~1 HEAD 2^>nul') do (
        if not "%%f"=="sync_to_internal.bat" (
            xcopy /Y /Q "D:\Hush\%%f" "D:\callbot\hush\%%f*" >nul 2>&1
            set /a count+=1
        )
    )
)

echo Synced !count! file(s).
endlocal
