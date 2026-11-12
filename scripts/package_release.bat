@echo off
rem Сборка релизного архива: build.bat (сборка+тесты) -> dist\llmhost-<ver>-win-x64.zip.
rem В архив идёт ТОЛЬКО нужное пользователю: exe, manifest, README, notes, лицензия.
setlocal
set "ROOT=%~dp0.."
set "REL=%ROOT%\build\Release"
set "VER=0.6.0-beta9"
set "NAME=llmhost-%VER%-win-x64"
set "DIST=%ROOT%\dist"

if not exist "%REL%\llmhost.exe" (echo нет сборки — запустите build.bat & exit /b 1)
if not exist "%REL%\lhctl.exe" (echo нет lhctl.exe & exit /b 1)

mkdir "%DIST%" 2>nul
set "STAGE=%DIST%\%NAME%"
mkdir "%STAGE%" 2>nul

copy /y "%REL%\llmhost.exe"     "%STAGE%" >nul || exit /b 1
copy /y "%REL%\lhctl.exe"       "%STAGE%" >nul || exit /b 1
copy /y "%REL%\manifest.json"   "%STAGE%" >nul || exit /b 1
copy /y "%ROOT%\README.md"      "%STAGE%" >nul || exit /b 1
copy /y "%ROOT%\RELEASE_NOTES.txt" "%STAGE%" >nul || exit /b 1
copy /y "%ROOT%\LICENSE"        "%STAGE%" >nul || exit /b 1

powershell -NoProfile -Command "Compress-Archive -Force -Path '%STAGE%\*' -DestinationPath '%DIST%\%NAME%.zip'"
if errorlevel 1 (echo упаковка не удалась & exit /b 1)
powershell -NoProfile -Command "(Get-FileHash '%DIST%\%NAME%.zip' -Algorithm SHA256).Hash" > "%DIST%\%NAME%.zip.sha256"

echo Готово: %DIST%\%NAME%.zip
echo SHA256:
type "%DIST%\%NAME%.zip.sha256"
endlocal
