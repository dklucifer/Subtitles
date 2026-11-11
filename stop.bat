@echo off
rem Stop the running Subtitles service cleanly.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Method POST -Uri http://127.0.0.1:8765/api/shutdown -TimeoutSec 5 | Out-Null; Write-Host 'Subtitles stopped' } catch { Write-Host 'Subtitles is not running' }"
