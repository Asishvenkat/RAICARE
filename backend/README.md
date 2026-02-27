# RAiCare Backend

**Complete FastAPI backend for RAiCare - Rheumatoid Arthritis Detection & Management System**

## 🚀 Features

### ✅ Implemented Features
- **User Authentication** - JWT-based registration and login
- **Image Upload** - X-ray upload to Cloudinary with URL storage
- **Prediction Storage** - Save RA detection results with severity levels
- **AI Chatbot** - Personalized recommendations using Google Gemini AI
- **Chat History** - Store and retrieve conversation history
- **MongoDB Integration** - NoSQL database for all data storage
- **CORS Support** - Ready for React frontend integration

### 🎯 API Endpoints

#### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login and get JWT token

#### Predictions
- `POST /prediction/upload` - Upload X-ray and save prediction
- `GET /prediction/history` - Get user's prediction history
- `GET /prediction/latest` - Get most recent prediction

#### Chatbot
- `POST /chat/send` - Send message to AI chatbot
- `GET /chat/history` - Get chat history
- `GET /chat/welcome` - Get personalized welcome message
- `DELETE /chat/clear` - Clear chat history

#### Health
- `GET /` - API information
- `GET /health` - Health check

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # MongoDB models & Pydantic schemas
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── prediction.py      # Prediction endpoints
│   │   └── chat.py            # Chatbot endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cloudinary_service.py   # Image upload service
│   │   └── chatbot_service.py      # AI chatbot logic
│   └── utils/
│       ├── __init__.py
│       ├── auth.py            # JWT utilities
│       └── database.py        # MongoDB connection
├── main.py                    # FastAPI application entry point
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── .gitignore
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- MongoDB Atlas account (already configured)
- Cloudinary account (already configured)
- Google Gemini API key (already configured)

### Step 1: Navigate to Backend Directory
```bash
cd backend
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Variables
The `.env` file is already created with your credentials. **IMPORTANT**: Generate a secure SECRET_KEY:

```bash
# Generate secure secret key (Windows PowerShell)
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output and update `SECRET_KEY` in `.env` file.

### Step 5: Run the Server
```bash
# Development mode with auto-reload
uvicorn main:app --reload

# Or using Python
python main.py
```

The server will start at: **http://localhost:8000**

### Step 6: Access API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔌 MongoDB Collections

### 1. **users**
```json
{
  "_id": "ObjectId",
  "username": "string",
  "email": "string",
  "hashed_password": "string",
  "created_at": "datetime"
}
```

### 2. **predictions**
```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "image_url": "string",
  "result_percentage": "float (0-100)",
  "severity_level": "mild | moderate | severe",
  "timestamp": "datetime"
}
```

### 3. **chat_history**
```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "message": "string",
  "response": "string",
  "timestamp": "datetime"
}
```

---

## 🤖 AI Chatbot Logic

### Personalized Recommendations Based on Severity

#### **MILD RA**
- **Diet**: Anti-inflammatory foods, omega-3, berries, leafy greens
- **Exercise**: Walking, swimming, yoga, tai chi (30 min/day)
- **Lifestyle**: Stress management, 7-8 hours sleep, healthy weight

#### **MODERATE RA**
- **Diet**: Mediterranean diet, strict anti-inflammatory foods, AVOID red meat, processed foods, sugar
- **Exercise**: Water aerobics, gentle cycling (20-30 min, 4-5 days/week)
- **Lifestyle**: Weight management critical, hot/cold therapy, ergonomic adjustments

#### **SEVERE RA**
- **Diet**: STRICT anti-inflammatory diet, plant-based recommended, ABSOLUTELY AVOID red meat, processed foods, alcohol
- **Exercise**: GENTLE range-of-motion only, water therapy, consult physical therapist
- **Lifestyle**: URGENT medical supervision, prioritize rest (8-9 hours), assistive devices, support groups

---

## 🔐 Authentication Flow

### Registration
```bash
POST /auth/register
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "securepass123"
}
```

### Login
```bash
POST /auth/login
{
  "email": "john@example.com",
  "password": "securepass123"
}

# Response includes JWT token
{
  "status": "success",
  "data": {
    "access_token": "eyJhbGc...",
    "token_type": "bearer",
    "user": {...}
  }
}
```

### Protected Endpoints
Include JWT token in header:
```
Authorization: Bearer <your_token_here>
```

---

## 📤 Image Upload Flow

1. User uploads X-ray image from React frontend
2. Image sent to `/prediction/upload` with `result_percentage` and `severity_level`
3. Backend uploads image to Cloudinary
4. Cloudinary URL saved in MongoDB with prediction data
5. Response sent back to frontend

**Example Request** (multipart/form-data):
```
POST /prediction/upload
Authorization: Bearer <token>

file: [image file]
result_percentage: 75.5
severity_level: moderate
```

---

## 💬 Chatbot Usage

### Send Message
```bash
POST /chat/send
Authorization: Bearer <token>
{
  "message": "What foods should I avoid?"
}

# Response personalized based on latest prediction
{
  "status": "success",
  "data": {
    "chat": {
      "message": "What foods should I avoid?",
      "response": "Based on your MODERATE RA condition...",
      "timestamp": "..."
    },
    "context": {
      "severity_level": "moderate",
      "result_percentage": 75.5
    }
  }
}
```

---

## 🌐 React Frontend Integration

### Setup Axios in React
```javascript
// src/api/axios.js
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  }
});

// Add token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
```

### Example API Calls

#### Register
```javascript
import api from './api/axios';

const register = async (userData) => {
  const response = await api.post('/auth/register', userData);
  return response.data;
};
```

#### Upload X-ray with Prediction
```javascript
const uploadXray = async (file, percentage, severity) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('result_percentage', percentage);
  formData.append('severity_level', severity);
  
  const response = await api.post('/prediction/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};
```

#### Chat with AI
```javascript
const sendChatMessage = async (message) => {
  const response = await api.post('/chat/send', { message });
  return response.data;
};
```

---

## 🧪 Testing the API

### Using cURL

#### Register User
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"test123"}'
```

#### Login
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'
```

### Using Swagger UI
Visit http://localhost:8000/docs for interactive API testing

---

## 📊 Response Format

All API responses follow this structure:
```json
{
  "status": "success | error",
  "message": "Description of the result",
  "data": {
    // Response data
  }
}
```

---

## 🔒 Security Features

- ✅ Password hashing with bcrypt
- ✅ JWT token authentication
- ✅ Token expiration (24 hours)
- ✅ Protected endpoints
- ✅ CORS configuration
- ✅ Input validation with Pydantic
- ✅ Secure environment variables

---

## 🚨 Troubleshooting

### MongoDB Connection Issues
- Verify MongoDB URL in `.env`
- Check network/firewall settings
- Ensure IP whitelist in MongoDB Atlas

### Cloudinary Upload Fails
- Verify Cloudinary credentials in `.env`
- Check image file size (max 10MB recommended)
- Ensure file format is JPG/PNG

### Gemini API Errors
- Verify API key in `.env`
- Check API quota/limits
- Ensure internet connectivity

---

## 📝 API Response Examples

### Successful Registration
```json
{
  "status": "success",
  "message": "User registered successfully",
  "data": {
    "user": {
      "id": "65abc123...",
      "username": "johndoe",
      "email": "john@example.com",
      "created_at": "2025-01-15T10:30:00"
    }
  }
}
```

### Prediction Upload
```json
{
  "status": "success",
  "message": "Prediction saved successfully",
  "data": {
    "prediction": {
      "id": "65abc456...",
      "user_id": "65abc123...",
      "image_url": "https://res.cloudinary.com/...",
      "result_percentage": 75.5,
      "severity_level": "moderate",
      "timestamp": "2025-01-15T11:00:00"
    }
  }
}
```

---

## 🎓 Production Deployment

### Environment Setup
1. Update `.env` with production values
2. Set `FRONTEND_URL` to production domain
3. Use strong `SECRET_KEY`
4. Enable HTTPS

### Recommended Hosting
- **Backend**: Railway, Render, AWS, DigitalOcean
- **Database**: MongoDB Atlas (already configured)
- **Images**: Cloudinary (already configured)

### Production Command
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 📞 Support

For issues or questions:
1. Check API documentation: http://localhost:8000/docs
2. Review error messages in console
3. Check MongoDB Atlas logs
4. Verify environment variables

---

## ✨ Next Steps

1. ✅ Start the backend server
2. ✅ Test endpoints using Swagger UI
3. ✅ Integrate with React frontend
4. ✅ Test authentication flow
5. ✅ Test image upload
6. ✅ Test chatbot functionality
7. ✅ Deploy to production

---

**Built with ❤️ using FastAPI, MongoDB, Cloudinary, and Google Gemini AI**
