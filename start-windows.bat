@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.12 3.13 3.11 3.10) do (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 (
            set "PHYSICS_PYTHON=py -%%V"
            goto run
        )
    )
)
where python >nul 2>nul
if errorlevel 1 (
    echo Install Python 3.12 from python.org and enable Add Python to PATH.
    pause
    exit /b 1
)
set "PHYSICS_PYTHON=python"
:run
%PHYSICS_PYTHON% launch.py %*
set "PHYSICS_RESULT=%ERRORLEVEL%"
if not "%PHYSICS_RESULT%"=="0" pause
exit /b %PHYSICS_RESULT%
