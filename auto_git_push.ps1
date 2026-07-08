$ErrorActionPreference = "SilentlyContinue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Auto Git Committer & Pusher (Watcher) " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop monitoring." -ForegroundColor Yellow
Write-Host ""

# 현재 브랜치 이름은 로깅용으로만 가져옵니다
$branch = "feature/gate-rm"

Write-Host "Watching repository for codebase changes..." -ForegroundColor Green

while ($true) {
    # 변경 사항 감지 (Untracked, Modified, Deleted 등)
    $status = git status --porcelain
    
    if ($status) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "`n[$timestamp] 🔍 Changes detected! Staging files..." -ForegroundColor Yellow
        
        # 변경 사항 스테이징
        git add .
        
        # 커밋
        Write-Host "[$timestamp] 💾 Committing changes..." -ForegroundColor Yellow
        git commit -m "chore(auto): codebase update at $timestamp" | Out-Null
        
        # 푸시
        Write-Host "[$timestamp] 🚀 Pushing to origin $branch..." -ForegroundColor Yellow
        $pushResult = git push origin $branch 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$timestamp] ✅ Push successful!" -ForegroundColor Green
        } else {
            Write-Host "[$timestamp] ❌ Push failed: $pushResult" -ForegroundColor Red
        }
    }
    
    # 30초 대기 후 다시 검사
    Start-Sleep -Seconds 30
}
