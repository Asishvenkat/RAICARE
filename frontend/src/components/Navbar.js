import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './Navbar.css';

function Navbar() {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isActive = (path) => {
    return location.pathname === path ? 'active' : '';
  };

  if (!isAuthenticated) {
    return null;
  }

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/dashboard" className="navbar-brand">
          <span className="brand-icon">🏥</span>
          <span className="brand-name">RAiCare</span>
        </Link>

        <div className="navbar-menu">
          <Link to="/dashboard" className={`nav-link ${isActive('/dashboard')}`}>
            Dashboard
          </Link>
          <Link to="/upload" className={`nav-link ${isActive('/upload')}`}>
            Upload
          </Link>
          <Link to="/chatbot" className={`nav-link ${isActive('/chatbot')}`}>
            AI Assistant
          </Link>
          <Link to="/history" className={`nav-link ${isActive('/history')}`}>
            History
          </Link>
        </div>

        <div className="navbar-user">
          <div className="user-info">
            <div className="user-avatar">{user && user.username ? user.username.charAt(0).toUpperCase() : '?'}</div>
            <span className="user-name">{user?.username}</span>
          </div>
          <button onClick={handleLogout} className="logout-btn">
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
