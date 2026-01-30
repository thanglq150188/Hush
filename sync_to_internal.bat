@echo off
:: Sync changed files from D:\Hush -> D:\callbot\hush
:: Works both before and after commit

cd /d D:\Hush
setlocal enabledelayedexpansion

set count=0

:: 1. Uncommitted changes (modified + staged)
for /f "delims=" %%f in ('git diff --name-only HEAD 2^>nul') do (
    if not "%%f"=="sync_to_internal.bat" (
        set "fp=%%f"
        set "fp=!fp:/=\!"
        copy /Y "D:\Hush\!fp!" "D:\callbot\hush\!fp!" 1>NUL 2>NUL
        if !errorlevel!==0 (
            echo   OK: %%f
            set /a count+=1
        ) else (
            echo   FAILED: %%f
        )
    )
)

:: 2. Staged files
for /f "delims=" %%f in ('git diff --name-only --cached 2^>nul') do (
    if not "%%f"=="sync_to_internal.bat" (
        set "fp=%%f"
        set "fp=!fp:/=\!"
        copy /Y "D:\Hush\!fp!" "D:\callbot\hush\!fp!" 1>NUL 2>NUL
        if !errorlevel!==0 (
            echo   OK: %%f
            set /a count+=1
        ) else (
            echo   FAILED: %%f
        )
    )
)

:: 3. Untracked new files
for /f "delims=" %%f in ('git ls-files --others --exclude-standard 2^>nul') do (
    if not "%%f"=="sync_to_internal.bat" (
        set "fp=%%f"
        set "fp=!fp:/=\!"
        copy /Y "D:\Hush\!fp!" "D:\callbot\hush\!fp!" 1>NUL 2>NUL
        if !errorlevel!==0 (
            echo   OK: %%f
            set /a count+=1
        ) else (
            echo   FAILED: %%f
        )
    )
)

:: 4. Last commit changes (for running after commit)
if !count!==0 (
    for /f "delims=" %%f in ('git diff --name-only HEAD~1 HEAD 2^>nul') do (
        if not "%%f"=="sync_to_internal.bat" (
            set "fp=%%f"
            set "fp=!fp:/=\!"
            copy /Y "D:\Hush\!fp!" "D:\callbot\hush\!fp!" 1>NUL 2>NUL
            if !errorlevel!==0 (
                echo   OK: %%f
                set /a count+=1
            ) else (
                echo   FAILED: %%f
            )
        )
    )
)

echo Synced !count! file(s).
endlocal
