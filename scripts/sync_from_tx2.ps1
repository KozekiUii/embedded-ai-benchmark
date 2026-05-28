# 从 TX2 NX 拉回运行产物到本机仓库（在本机 PowerShell 执行）
# 用法: .\scripts\sync_from_tx2.ps1
# 只回流会变化的产物：results/（benchmark 数据+图） 和 根目录的标注图 out_*.jpg
# engine/onnx 体积大且设备端现编，默认不拉（需要时手动 scp）

$ErrorActionPreference = "Stop"
$Remote   = "tx2"   # ssh 直连别名（169.254.67.100，走网线）
$RemoteDir = "~/embedded-ai-benchmark"
$LocalDir = Split-Path -Parent $PSScriptRoot   # 仓库根目录

Write-Host "[sync] pulling results/ ..." -ForegroundColor Cyan
scp -q -r "${Remote}:${RemoteDir}/results" "$LocalDir/"

Write-Host "[sync] pulling annotated images out_*.jpg ..." -ForegroundColor Cyan
# 远端可能没有匹配文件，忽略报错
scp -q "${Remote}:${RemoteDir}/out_*.jpg" "$LocalDir/" 2>$null

Write-Host "[sync] done -> $LocalDir" -ForegroundColor Green
