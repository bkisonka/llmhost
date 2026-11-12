@echo off
rem Сборка llmhost: нужен VS 2022 Build Tools (C++ + CMake).
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul || (echo vcvars64.bat не найден & exit /b 1)
cmake -S "%ROOT%" -B "%ROOT%\build" -G "Visual Studio 17 2022" -A x64 || exit /b 1
cmake --build "%ROOT%\build" --config Release || exit /b 1
cd /d "%ROOT%\build"
ctest -C Release --output-on-failure
echo.
echo Готово: %ROOT%\build\Release\llmhost.exe
