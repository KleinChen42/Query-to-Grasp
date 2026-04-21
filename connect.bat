@echo off
:: 双击打开 H200 交互终端
:: 如需落在指定目录：改下面一行，加上 -RemoteDir "~/YourProject"
powershell -ExecutionPolicy Bypass -NoExit -File "%~dp0connect.ps1"
