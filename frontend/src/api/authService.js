import api from './client';

export const authService = {
  // Register new user
  register: async (username, email, password) => {
    const response = await api.post('/auth/register', {
      username,
      email,
      password,
    });
    return response.data;
  },

  // Login user
  login: async (email, password) => {
    const response = await api.post('/auth/login', {
      email,
      password,
    });
    
    if (response.data.status === 'success') {
      const { access_token, user } = response.data.data;
      
      // Store token and user in localStorage
      localStorage.setItem('authToken', access_token);
      localStorage.setItem('user', JSON.stringify(user));
    }
    
    return response.data;
  },

  // Logout
  logout: () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
  },

  // Get current user
  getCurrentUser: () => {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  },

  // Check if user is authenticated
  isAuthenticated: () => {
    return !!localStorage.getItem('authToken');
  },
};
