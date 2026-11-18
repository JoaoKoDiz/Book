@echo off
REM ---------------------------------------------------------------------------
REM  Runs (or resumes) the complete reconstruction with the files downloaded by
REM  "python -m badapple_dr download --out sources".  Safe to stop (Ctrl+C) and
REM  start again: finished work is reused.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first & exit /b 1)

set REF=
for %%F in (sources\badapple_reference.*) do if /I not "%%~xF"==".json" if /I not "%%~xF"==".part" if /I not "%%~xF"==".ytdl" if /I not "%%~xF"==".temp" set REF=%%F
set SRC=
for %%F in (sources\deltarune_480p.*) do if /I not "%%~xF"==".json" if /I not "%%~xF"==".part" if /I not "%%~xF"==".ytdl" if /I not "%%~xF"==".temp" set SRC=%%F
set HQ=
for %%F in (sources\deltarune_720p30.*) do if /I not "%%~xF"==".json" if /I not "%%~xF"==".part" if /I not "%%~xF"==".ytdl" if /I not "%%~xF"==".temp" set HQ=%%F

if "%REF%"=="" (echo sources\badapple_reference.* not found & exit /b 1)
if "%SRC%"=="" (echo sources\deltarune_480p.* not found & exit /b 1)

set EXTRA=
if not "%HQ%"=="" set EXTRA=--export-source "%HQ%"

python -m badapple_dr run --reference "%REF%" --source "%SRC%" %EXTRA% --work work --out output %*
endlocal
