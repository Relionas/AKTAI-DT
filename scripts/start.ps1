# Запуск сервиса на Windows: .\scripts\start.ps1  (из корня проекта)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path ".env")) { Write-Host "Нет файла .env. Скопируйте .env.example в .env и задайте API_KEY." -ForegroundColor Red; exit 1 }
if (Test-Path ".\.venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }
# Не даём компьютеру уснуть, пока идёт проверка жюри (питание от сети)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
python backend\api.py
