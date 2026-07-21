@echo off
setlocal enabledelayedexpansion

REM Frog Spectrogram Labeler launcher (Windows)
REM - Double-click to run
REM - Creates .venv on first run
REM - Installs locked dependencies from requirements_labeler.txt
REM - Launches Streamlit UI

cd /d "%~dp0"

echo ==========================================
echo Frog Spectrogram Labeler - Launcher
echo Repo: %CD%
echo ==========================================
echo.

set VENV_DIR=.venv
set PYEXE=%VENV_DIR%\Scripts\python.exe

REM Pick Python 3.13 if the venv does not exist yet.
set SYS_PY=
where py >nul 2>nul && py -3.13 -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 13))" >nul 2>nul && set SYS_PY=py -3.13
if "%SYS_PY%"=="" (
  where python >nul 2>nul && python -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 13))" >nul 2>nul && set SYS_PY=python
)

if not exist "%PYEXE%" (
  echo [SETUP] Creating virtual environment in "%VENV_DIR%"...
  if "%SYS_PY%"=="" (
    echo [ERROR] Python 3.13 was not found. Please install Python 3.13 from python.org and try again.
    echo.
    pause
    exit /b 1
  )
  %SYS_PY% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    echo.
    pause
    exit /b 1
  )
)

"%PYEXE%" -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 13))" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] The existing .venv does not use Python 3.13.
  echo Remove .venv, then run this launcher again to recreate it.
  echo.
  pause
  exit /b 1
)

echo [SETUP] Installing locked dependencies...
"%PYEXE%" -m pip install --upgrade pip >nul
"%PYEXE%" -m pip install -r requirements_labeler.txt
if errorlevel 1 (
  echo.
  echo [ERROR] Dependency install failed.
  echo Tip: check your internet connection, then re-run this launcher.
  echo.
  pause
  exit /b 1
)

echo.
echo [RUN] Starting the labeling UI...
echo Close this window to stop the app.
echo.

REM Streamlit opens your browser automatically.
"%PYEXE%" -m streamlit run "scripts\label_frontend.py" --server.address localhost --server.port 8501

echo.
echo [DONE] Streamlit stopped.
pause

