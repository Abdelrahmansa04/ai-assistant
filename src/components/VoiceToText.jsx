import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import './VoiceToText.css';
import io from 'socket.io-client';
import VoiceChat from './VoiceChat';

const WEBHOOK_URL = 'https://to7a3.app.n8n.cloud/webhook/f17e458d-9059-42c2-8d14-57acda06fc41';
const SOCKET_URL = 'http://localhost:5000';

const VoiceToText = () => {
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [mode, setMode] = useState('text'); // 'text' or 'voice'
  const speechSynthesis = window.speechSynthesis;

  const [chatImages, setChatImages] = useState([]);
  const [selectedImages, setSelectedImages] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const voices = useRef([]);

  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  useEffect(() => {
    // Load chat history from localStorage on component mount
    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
      setChatHistory(JSON.parse(savedHistory));
    }
  }, []);

  useEffect(() => {
    const newSocket = io(SOCKET_URL, {
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });

    newSocket.on('connect', () => setIsConnected(true));
    newSocket.on('disconnect', () => setIsConnected(false));
    newSocket.on('n8n-message', handleAIResponse);

    setSocket(newSocket);
    return () => newSocket.disconnect();
  }, []);

  useEffect(() => {
    // Load available voices
    const loadVoices = () => {
      voices.current = speechSynthesis.getVoices();
    };

    loadVoices();
    if (speechSynthesis.onvoiceschanged !== undefined) {
      speechSynthesis.onvoiceschanged = loadVoices;
    }

    return () => {
      if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
      }
    };
  }, []);

  const speak = (text) => {
    if (speechSynthesis.speaking) {
      speechSynthesis.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(text);
    // Try to find an English voice
    const englishVoice = voices.current.find(voice =>
      voice.lang.startsWith('en') && voice.name.includes('Male')
    );
    if (englishVoice) {
      utterance.voice = englishVoice;
    }
    utterance.rate = 1;
    utterance.pitch = 1;
    speechSynthesis.speak(utterance);
  };

  const handleAIResponse = useCallback((data) => {
    const messageText = typeof data.message === 'string'
      ? data.message
      : JSON.stringify(data.message, null, 2);

    addMessage('ai', messageText);

    if (mode === 'voice') {
      speak(messageText);
    }
  }, [mode]);

  const saveCurrentChatToHistory = () => {
    if (messages.length > 0) {
      const newChat = {
        id: Date.now(),
        messages: [...messages],
        timestamp: new Date().toLocaleString(),
        preview: messages[0].text.slice(0, 50) + '...'
      };

      setChatHistory(prev => {
        const updatedHistory = [newChat, ...prev];
        localStorage.setItem('chatHistory', JSON.stringify(updatedHistory));
        return updatedHistory;
      });
    }
  };

  const startNewChat = () => {
    saveCurrentChatToHistory();
    setMessages([]);
    setInputText('');
    if (listening) {
      resetTranscript();
      SpeechRecognition.stopListening();
    }
  };

  const loadChatFromHistory = (chatId) => {
    const selectedChat = chatHistory.find(chat => chat.id === chatId);
    if (selectedChat) {
      saveCurrentChatToHistory();
      setMessages(selectedChat.messages);
      setShowHistory(false);
    }
  };


  /////////////////////////////
  const handleFileSelect = (files) => {
    const imageFiles = Array.from(files).filter(file =>
      file.type.startsWith('image/')
    );

    if (imageFiles.length === 0) {
      setError('Please select valid image files (PNG, JPG, GIF, etc.)');
      return;
    }

    const newImages = imageFiles.map(file => ({
      id: Date.now() + Math.random(),
      file,
      preview: URL.createObjectURL(file),
      name: file.name
    }));

    setChatImages(prev => [...prev, ...newImages]);
    setSelectedImages(prev => [...prev, ...newImages]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const removeImage = (imageId) => {
    setSelectedImages(prev => {
      const updated = prev.filter(img => img.id !== imageId);
      // Clean up object URLs
      const removed = prev.find(img => img.id === imageId);
      if (removed) {
        URL.revokeObjectURL(removed.preview);
      }
      return updated;
    });
  };
  /////////////////////////////


  ///////////////////////
  const handleSendMessage = async (text) => {
    if (!text.trim() || isProcessing) return;

    try {
      setIsProcessing(true);
      addMessage('user', text);

      await axios.post(WEBHOOK_URL, { message: text }, {
        headers: { 'Content-Type': 'application/json' }
      });

      setInputText('');
      setSelectedImages([]);
      if (listening) {
        SpeechRecognition.stopListening();
        resetTranscript();
      }
    } catch (error) {
      setError('Failed to send message. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };
  /////////////////////////


  /////////////////////////////
  const handleSendImage = async () => {
    if ((selectedImages.length === 0) || isProcessing) return;

    try {
      setIsProcessing(true);

      const messageText = 'Image(s) sent';
      const messageImages = selectedImages.map(img => ({
        preview: img.preview,
        name: img.name
      }));
      addMessage('user', messageText, messageImages);

      let requestData;
      let headers;

      if (selectedImages.length > 0) {
        const formData = new FormData();
        formData.append('body', 'Image(s) attached');
        selectedImages.forEach((image, index) => {
          formData.append(`image_${index}`, image.file);
        });
        requestData = formData;
        headers = {}; // Browser sets Content-Type for FormData
      }
      // else {
      //   requestData = { body: text };
      //   headers = { 'Content-Type': 'application/json' };
      // }

      await axios.post("http://localhost:5678/yolo", requestData, { headers });

      setInputText('');
      setSelectedImages([]);
      if (listening) {
        SpeechRecognition.stopListening();
        resetTranscript();
      }

    } catch (error) {
      console.error('Webhook Error:', error.response?.data || error.message);
      setError('Failed to send message. Please check server.');
    } finally {
      setIsProcessing(false);
    }
  };
  /////////////////////////////

  const handleVoiceInput = useCallback(() => {
    if (!listening) {
      setInputText('');
      resetTranscript();
      SpeechRecognition.startListening({ continuous: true });
    } else {
      SpeechRecognition.stopListening();
      setInputText(transcript);
    }
  }, [listening, transcript]);

  useEffect(() => {
    if (listening && transcript) {
      setInputText(transcript);
    }
  }, [transcript, listening]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(inputText);
    }
  };

  const addMessage = (sender, text, images = []) => {
    setMessages(prev => [...prev, {
      id: Date.now(),
      sender,
      text,
      images,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);
  };

  const switchMode = () => {
    if (listening) {
      SpeechRecognition.stopListening();
    }
    setMode(mode === 'text' ? 'voice' : 'text');
  };


  /////////////////////////////
  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      selectedImages.forEach(img => {
        URL.revokeObjectURL(img.preview);
      });
    };
  }, []);
  /////////////////////////////



  if (mode === 'voice') {
    return <VoiceChat onSwitchMode={switchMode} />;
  }

  if (!browserSupportsSpeechRecognition) {
    return (
      <div className="error-screen">
        <p>Browser doesn't support speech recognition. Please use Chrome or Edge.</p>
      </div>
    );
  }

  return (
    <div
      className={`chat-app ${isDragging ? 'dragging' : ''}`} /////////////////////////////
      onDragOver={handleDragOver} /////////////////////////////
      onDragLeave={handleDragLeave} /////////////////////////////
      onDrop={handleDrop} /////////////////////////////
    >
      <header className="chat-header">
        <h1>AI Assistant</h1>
        <div className="header-actions">
          <button
            className="mode-switch"
            onClick={switchMode}
            title="Switch to voice chat"
          >
            <i className="fas fa-microphone" />
          </button>
          <div className={`connection-status ${isConnected ? 'connected' : ''}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
          <button
            className="history-button"
            onClick={() => setShowHistory(true)}
            title="View chat history"
          >
            <i className="fas fa-history" />
          </button>
        </div>
      </header>

      <main className="chat-main">
        <div className="messages">
          {messages.map(({ id, sender, text, images, time }) => (
            <div key={id} className={`message ${sender}`}>
              {/* start Image selection */}
              {images && images.length > 0 && (
                <div className="message-images">
                  {images.map((image, index) => (
                    <div key={index} className="message-image">
                      <img className='message-img' src={image.preview} alt={image.name} />
                      <span className="image-name">{image.name}</span>
                    </div>
                  ))}
                </div>
              )}
              {/* end Image selection */}

              <p className='message-text'>{text}</p>
              <time>{time}</time>
            </div>
          ))}


          {isProcessing && (
            <div className="message ai typing">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
        </div>

        <div className="input-area">
          {error && <div className="error" onClick={() => setError(null)}>{error}</div>}

          {/* start Image selection */}
          {selectedImages.length > 0 && (
            <div className="selected-images">
              {selectedImages.map((image) => (
                <div key={image.id} className="selected-image">
                  <img className='selected-img' src={image.preview} alt={image.name} />
                  
                  <span className="image-name">{image.name}</span>
                  <button
                    className="remove-image"
                    onClick={() => removeImage(image.id)}
                    title="Remove image"
                  >
                    <i className="fas fa-times"></i>
                  </button>
                </div>
              ))}
            </div>
          )}
          {/* end Image selection */}


          <div className="input-container">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a message..."
              disabled={isProcessing}
              rows={1}
            />

            <div className="actions">

              {/* start Image input */}
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => handleFileSelect(e.target.files)}
                accept="image/*"
                multiple
                style={{ display: 'none' }}
              />
              {/* end Image Input */}

              {/* start Image upload */}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="image-upload"
                disabled={isProcessing}
                title="Upload images"
              >
                <i className="fas fa-image" />
              </button>
              {/* end Image upload */}

              {/* start Image send */}
              <button
                onClick={() => handleSendImage()}
                className="image-send"
                disabled={isProcessing}
                title="Send images"
              >
                YOLO
              </button>
              {/* end Image send */}

              <button
                onClick={handleVoiceInput}
                className={`voice ${listening ? 'active' : ''}`}
                disabled={isProcessing}
                title={listening ? "Stop recording" : "Start recording"}
              >
                <i className={`fas ${listening ? 'fa-stop' : 'fa-microphone'}`} />
              </button>

              <button
                onClick={startNewChat}
                className="new-chat-button"
                title="Start new chat"
                disabled={isProcessing}
              >
                <i className="fas fa-plus-circle" />
              </button>

              <button
                onClick={() => handleSendMessage(inputText)}
                className="send"
                disabled={(!inputText.trim() && !transcript) || isProcessing}
                title="Send message"
              >
                <i className="fas fa-paper-plane" />
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* start Drag overlay */}
      {isDragging && (
        <div className="drag-overlay">
          <div className="drag-content">
            <i className="fas fa-cloud-upload-alt"></i>
            <p>Drop images here to upload</p>
          </div>
        </div>
      )}
      {/* end Drag overlay */}

      {showHistory && (
        <div className="modal-overlay">
          <div className="modal history-modal">
            <div className="history-header">
              <h3>
                <i className="fas fa-history"></i>
                Chat History
              </h3>
              <button
                onClick={() => setShowHistory(false)}
                className="close-button"
                title="Close history"
              >
                <i className="fas fa-times"></i>
              </button>
            </div>

            <div className="chat-history-list">
              {chatHistory.length > 0 ? (
                chatHistory.map(chat => (
                  <div
                    key={chat.id}
                    className="history-item"
                    onClick={() => loadChatFromHistory(chat.id)}
                  >
                    <div className="history-item-content">
                      <div className="history-item-header">
                        <span className="message-count">
                          <i className="fas fa-comments"></i>
                          {chat.messages.length} messages
                        </span>
                        <time>
                          <i className="fas fa-clock"></i>
                          {chat.timestamp}
                        </time>
                      </div>
                      <p className="preview">{chat.preview}</p>
                    </div>
                    <i className="fas fa-chevron-right history-arrow"></i>
                  </div>
                ))
              ) : (
                <div className="no-history">
                  <i className="fas fa-inbox"></i>
                  <p>No chat history yet</p>
                  <span>Your chat history will appear here</span>
                </div>
              )}
            </div>

            <div className="history-footer">
              <button
                onClick={() => setShowHistory(false)}
                className="secondary-button"
              >
                Close
              </button>
              {chatHistory.length > 0 && (
                <button
                  onClick={() => {
                    setChatHistory([]);
                    localStorage.removeItem('chatHistory');
                    setShowHistory(false);
                  }}
                  className="danger-button"
                >
                  <i className="fas fa-trash-alt"></i>
                  Clear History
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceToText;
