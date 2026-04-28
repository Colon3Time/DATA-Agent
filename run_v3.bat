@echo off
cd /d %~dp0
powershell -NoExit -ExecutionPolicy Bypass -NoProfile -Command "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); $env:PYTHONUTF8='1'; cd '%~dp0'; python orchestrator_v3.py --no-color --no-title"
