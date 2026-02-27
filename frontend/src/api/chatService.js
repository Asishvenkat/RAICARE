import api from './client';

export const chatService = {
  // Send message to chatbot
  sendMessage: async (message) => {
    const response = await api.post('/chat/send', { message });
    return response.data;
  },

  // Get chat history
  getHistory: async (limit = 20) => {
    const response = await api.get(`/chat/history?limit=${limit}`);
    return response.data;
  },

  // Get welcome message
  getWelcome: async () => {
    const response = await api.get('/chat/welcome');
    return response.data;
  },

  // Clear chat history
  clearHistory: async () => {
    const response = await api.delete('/chat/clear');
    return response.data;
  },
};
