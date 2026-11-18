@echo off
REM ---------------------------------------------------------------------------
REM  One-time setup on Windows 11: creates .venv and installs the dependencies.
REM  Run from this folder (double-click or "setup_windows.bat" in a terminal).
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set PY=py -3
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install it with:  winget install --id Python.Python.3.12 -e
        echo then open a NEW terminal and run this script again.
        exit /b 1
    )
    set PY=python
)

if not exist .venv (
    echo Creating virtual environment .venv ...
    %PY% -m venv .venv || exit /b 1
)
call .venv\Scripts\activate.bat || exit /b 1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || exit /b 1

if /I "%1"=="semantic" (
    echo Installing optional semantic-feature packages ^(torch + transformers^) ...
    python -m pip install -r requirements-semantic.txt || exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo.
    echo NOTE: FFmpeg is not on PATH. The bundled copy from imageio-ffmpeg will be used.
    echo       For ffprobe and best speed install FFmpeg:  winget install --id Gyan.FFmpeg -e
)
where deno >nul 2>nul
if errorlevel 1 (
    where node >nul 2>nul
    if errorlevel 1 (
        echo.
        echo NOTE: YouTube downloads need a JavaScript runtime. Install Deno:  winget install DenoLand.Deno
    )
)
echo.
echo Setup finished. Next:
echo     .venv\Scripts\activate
echo     python -m badapple_dr selftest          ^(optional, ~5-10 min, validates the software^)
echo     python -m badapple_dr download --out sources
echo     run_windows.bat
endlocal
