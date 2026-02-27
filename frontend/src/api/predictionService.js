import api from './client';

export const predictionService = {
  // Upload X-ray and get AI prediction
  uploadPrediction: async (imageFile) => {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await api.post('/prediction/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  // Get prediction history
  getHistory: async (limit = 10) => {
    const response = await api.get(`/prediction/history?limit=${limit}`);
    return response.data;
  },

  // Get latest prediction
  getLatest: async () => {
    const response = await api.get('/prediction/latest');
    return response.data;
  },
};
