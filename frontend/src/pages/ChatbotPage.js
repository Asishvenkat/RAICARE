import React, { useState, useEffect, useRef } from 'react';
import { chatService } from '../api/chatService';
import LoadingSpinner from '../components/LoadingSpinner';
import './ChatbotPage.css';

function ChatbotPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    loadWelcomeAndHistory();
  }, []);

  const loadWelcomeAndHistory = async () => {
    try {
      // Load welcome message
      const welcomeResponse = await chatService.getWelcome();
      if (welcomeResponse.status === 'success') {
        setMessages([
          {
            type: 'bot',
            text: welcomeResponse.data.message,
            timestamp: new Date(),
          },
        ]);
      }

      // Load chat history
      const historyResponse = await chatService.getHistory(10);
      if (historyResponse.status === 'success' && historyResponse.data.chats.length > 0) {
        const history = historyResponse.data.chats.reverse().map((chat) => [
          { type: 'user', text: chat.message, timestamp: new Date(chat.timestamp) },
          { type: 'bot', text: chat.response, timestamp: new Date(chat.timestamp) },
        ]).flat();
        
        setMessages((prev) => [...prev, ...history]);
      }
    } catch (error) {
      console.error('Failed to load chat:', error);
      setMessages([
        {
          type: 'bot',
          text: '👋 Welcome to RAiCare AI Assistant! I can help you with personalized recommendations for managing RA. Please upload an X-ray first to get started.',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setInitialLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      type: 'user',
      text: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await chatService.sendMessage(input);
      
      if (response.status === 'success') {
        const botMessage = {
          type: 'bot',
          text: response.data.chat.response,
          timestamp: new Date(response.data.chat.timestamp),
        };
        setMessages((prev) => [...prev, botMessage]);
      }
    } catch (error) {
      const errorMessage = {
        type: 'bot',
        text: error.response?.data?.detail || 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const quickQuestions = [
    'What foods should I avoid?',
    'What exercises can I do?',
    'Tell me about lifestyle changes',
    'How can I manage pain?',
  ];

  const handleQuickQuestion = (question) => {
    setInput(question);
  };

  if (initialLoading) {
    return (
      <div className="chatbot-page">
        <div className="loading-container">
          <LoadingSpinner />
          <p>Loading your personalized assistant...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chatbot-page">
      <div className="chatbot-container">
        <div className="chat-header">
          <div className="header-content">
            <h1>🤖 RAiCare AI Assistant</h1>
            <p>Personalized recommendations based on your RA condition</p>
          </div>
        </div>

        <div className="messages-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.type}`}>
              <div className="message-avatar">
                {msg.type === 'bot' ? '🤖' : '👤'}
              </div>
              <div className="message-content">
                <div className="message-text">
                  {msg.text.split('\n').map((line, i) => (
                    <React.Fragment key={i}>
                      {line}
                      {i < msg.text.split('\n').length - 1 && <br />}
                    </React.Fragment>
                  ))}
                </div>
                <div className="message-time">
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="message bot">
              <div className="message-avatar">🤖</div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {messages.length === 1 && (
          <div className="quick-questions">
            <p>Try asking:</p>
            <div className="questions-grid">
              {quickQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => handleQuickQuestion(question)}
                  className="quick-question-btn"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="input-area">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about diet, exercise, lifestyle, or any RA-related questions..."
            rows="1"
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>
            {loading ? '...' : '➤'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatbotPage;
