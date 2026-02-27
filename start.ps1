# Start RAiCare Application (Frontend + Backend)

Write-Host "Starting RAiCare Application..." -ForegroundColor Green

# Start Backend
Write-Host "Starting Backend Server..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    Set-Location "E:\RA-Project\ra-detection\backend"
    python main.py
}

# Start Frontend
Write-Host "Starting Frontend Server..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    Set-Location "E:\RA-Project\ra-detection\frontend"
    npm start
}

Write-Host "Both servers are starting in the background!" -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "To stop, close this terminal or use Ctrl+C" -ForegroundColor Red

# Wait for jobs (optional, or keep running)
Wait-Job -Job $backendJob, $frontendJob