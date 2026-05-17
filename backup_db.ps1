# backup_db.ps1 — Erstellt ein timestamped Backup von training.db
# Aufruf: .\backup_db.ps1
# Backups werden in data/backups/ gespeichert und nicht in Git verfolgt.

$root   = $PSScriptRoot
$src    = Join-Path $root "data\training.db"
$outDir = Join-Path $root "data\backups"

if (-not (Test-Path $src)) {
    Write-Error "Datenbank nicht gefunden: $src"
    exit 1
}

if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$ts  = Get-Date -Format "yyyyMMdd_HHmmss"
$dst = Join-Path $outDir "training_$ts.db"

Copy-Item $src $dst
$sizeKB = [math]::Round((Get-Item $dst).Length / 1KB, 1)
Write-Host "Backup erstellt: $dst ($sizeKB KB)"

# Alte Backups auflisten
$all = Get-ChildItem $outDir -Filter "training_*.db" | Sort-Object Name
Write-Host "$($all.Count) Backup(s) vorhanden:"
$all | ForEach-Object { Write-Host "  $($_.Name)  ($([math]::Round($_.Length/1KB,1)) KB)" }
