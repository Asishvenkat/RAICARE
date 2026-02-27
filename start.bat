@echo off
echo Starting RAiCare Application...
echo.

echo Starting Backend Server...
start "Backend" cmd /k "cd /d E:\RA-Project\ra-detection\backend && python main.py"

timeout /t 2 /nobreak > nul

echo Starting Frontend Server...
start "Frontend" cmd /k "cd /d E:\RA-Project\ra-detection\frontend && npm start"

echo.
echo Both servers are starting!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo Close the opened command windows to stop the servers.

pause