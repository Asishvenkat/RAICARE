# Backend Server Setup Guide

## ✅ Status
The backend server is now **RUNNING** on `http://localhost:8000`

## Start Backend Server

### Windows PowerShell
```powershell
cd e:\RA-Project\ra-detection\backend
E:/RA-Project/.venv/Scripts/python.exe main.py
```

### Windows Command Prompt
```cmd
cd e:\RA-Project\ra-detection\backend
E:\RA-Project\.venv\Scripts\python.exe main.py
```

### Or using the batch file from root
```bash
cd e:\RA-Project\ra-detection
start.bat
```

## 🚀 API Endpoints Available

### Authentication Routes
- `POST /auth/register` - Register a new user
- `POST /auth/login` - Login user
- `GET /auth/me` - Get current user info

### Prediction Routes
- `POST /prediction/upload` - Upload X-ray and get AI prediction
- `GET /prediction/latest` - Get latest prediction
- `GET /prediction/history?limit=10` - Get prediction history

### Chat Routes
- `POST /chat` - Chat with AI assistant
- `GET /chat/history?limit=10` - Get chat history

## 📋 Requirements Met
- ✅ FastAPI server running
- ✅ MongoDB database connected
- ✅ CORS enabled for frontend
- ✅ JWT authentication ready
- ✅ Image prediction service loaded

## Frontend Configuration
The React frontend is configured to connect to:
```
http://localhost:8000
```

This is set via the environment variable `REACT_APP_API_URL` or defaults to `http://localhost:8000`.

## Common Issues & Solutions

### 1. **Connection Refused (net::ERR_CONNECTION_REFUSED)**
   - **Cause**: Backend server is not running
   - **Solution**: Start the backend server using commands above

### 2. **500 Internal Server Error on /prediction/upload**
   - **Cause**: Backend service error
   - **Solution**: Check backend console for error logs

### 3. **No Predictions Found**
   - **Cause**: User has no prediction history
   - **Solution**: Upload a new X-ray image first

### 4. **JWT Token Invalid**
   - **Cause**: Token expired or not sent properly
   - **Solution**: Login again to get a new token

## 🔍 Check Backend Logs

Open another terminal and monitor the backend logs:
```powershell
Get-Content $((Get-Process python).ProcessName | Select-Object -First 1) -Wait
```

Or simply check the running terminal where the server was started.

## 📦 Dependencies

All required packages are listed in `backend/requirements.txt`:
- FastAPI 0.109.0
- Uvicorn 0.27.0
- Motor 3.3.2 (MongoDB async driver)
- PyTorch 2.6.0+
- Google Gemini API

## 🌐 Access Points

- **Backend API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Frontend**: http://localhost:3000 (React dev server)

## Next Steps

1. ✅ Backend server is running
2. Start your React frontend:
   ```powershell
   cd e:\RA-Project\ra-detection\frontend
   npm start
   ```
3. Open http://localhost:3000 in your browser
4. Register/Login to access the application
5. Upload X-ray images for RA detection

