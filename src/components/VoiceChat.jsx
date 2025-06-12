import React, { useState, useEffect, useCallback, useRef } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import axios from 'axios';
import io from 'socket.io-client';
import './VoiceChat.css';

// const WEBHOOK_URL = 'https://to7a3.app.n8n.cloud/webhook/f17e458d-9059-42c2-8d14-57acda06fc41';
const WEBHOOK_URL = 'https://to7a3.app.n8n.cloud/webhook/f17e458d-9059-42c2-8d14-57acda06fc41';
const SOCKET_URL = 'http://localhost:5000';

const VoiceChat = ({ onSwitchMode }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const speechSynthesis = window.speechSynthesis;
  const voices = useRef([]);

  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  useEffect(() => {
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

  const speak = (text) => {
    if (speechSynthesis.speaking) {
      speechSynthesis.cancel();
    }
    
    SpeechRecognition.stopListening();

    const utterance = new SpeechSynthesisUtterance(text);
    const englishVoice = voices.current.find(voice => 
      voice.lang.startsWith('en') && voice.name.includes('Male')
    );
    if (englishVoice) {
      utterance.voice = englishVoice;
    }
    utterance.rate = 1;
    utterance.pitch = 1;

    utterance.onstart = () => {
      setIsSpeaking(true);
      SpeechRecognition.stopListening(); // Ensure listening stops as soon as speaking starts
    };

    
    utterance.onend = () => {
      setIsSpeaking(false);
      // Start listening again after AI finishes speaking
      setTimeout(() => {
        // if (!listening && !isProcessing) {
          SpeechRecognition.startListening({ continuous: true });
        // }
      }, 2000);
    };

    speechSynthesis.speak(utterance);
  };

  const handleAIResponse = useCallback((data) => {
    const messageText = typeof data.message === 'string' 
      ? data.message 
      : JSON.stringify(data.message, null, 2);

    addMessage('ai', messageText);
    speak(messageText);
  }, []);

  const addMessage = (sender, text) => {
    setMessages(prev => [...prev, {
      id: Date.now(),
      sender,
      text,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);
  };

  const handleSendMessage = async (text) => {
    if (!text.trim() || isProcessing) return;

    try {
      setIsProcessing(true);
      // Stop listening while processing
      if (listening) {
        SpeechRecognition.stopListening();
      }
      
      addMessage('user', text);
      resetTranscript();
      
      await axios.post(WEBHOOK_URL, { message: text }, {
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (error) {
      setError('Failed to send message. Please try again.');
      // Restart listening if there's an error
      if (!listening && !speechSynthesis.speaking) {
        SpeechRecognition.startListening({ continuous: true });
      }
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    // Auto-send message when user stops speaking
    let timeout;
    if (transcript && !isProcessing && !isSpeaking) {
      timeout = setTimeout(() => {
        handleSendMessage(transcript);
      }, 1500); // 1.5 second delay
    }

    return () => clearTimeout(timeout);
  }, [transcript, isProcessing, isSpeaking]);

  // Start listening automatically when component mounts
  useEffect(() => {
    if (!listening && !isProcessing && !isSpeaking) {
      SpeechRecognition.startListening({ continuous: true });
    }
  }, [listening, isProcessing, isSpeaking]);

  if (!browserSupportsSpeechRecognition) {
    return (
      <div className="error-screen">
        <p>Browser doesn't support speech recognition. Please use Chrome or Edge.</p>
      </div>
    );
  }

  return (
    <div className="voice-chat">
      <header className="voice-chat-header">
        <button 
          className="mode-switch" 
          onClick={onSwitchMode}
          title="Switch to text mode"
        >
          <i className="fas fa-keyboard" />
        </button>
        <h1>Voice Chat</h1>
        <div className="status-indicators">
          <div className={`connection-status ${isConnected ? 'connected' : ''}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
          <div className={`voice-status ${listening ? 'listening' : ''} ${isSpeaking ? 'speaking' : ''}`}>
            {isSpeaking ? 'AI Speaking' : (listening ? 'Listening...' : 'Click to start')}
          </div>
        </div>
      </header>

      <main className="voice-chat-main">
        <div className="messages">
          {messages.map(({ id, sender, text, time }) => (
            <div key={id} className={`message ${sender}`}>
              <p>{text}</p>
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

        <div className="voice-controls">
          <div className="voice-indicator">
            {transcript && <p className="transcript">{transcript}</p>}
            <div className="status-text">
              {isProcessing ? 'Processing...' : 
               isSpeaking ? 'AI is speaking...' : 
               listening ? 'Listening...' : 'Click microphone to start'}
            </div>
          </div>
          
          <button
            className={`voice-button ${listening ? 'active' : ''} ${isSpeaking ? 'speaking' : ''}`}
            onClick={() => {
              if (isSpeaking) {
                speechSynthesis.cancel();
                setIsSpeaking(false);
              } else if (listening) {
                SpeechRecognition.stopListening();
              } else {
                SpeechRecognition.startListening({ continuous: true });
              }
            }}
            disabled={isProcessing}
          >
            <i className={`fas ${isSpeaking ? 'fa-volume-up' : (listening ? 'fa-stop' : 'fa-microphone')}`} />
          </button>
        </div>
      </main>
    </div>
  );
};

export default VoiceChat; 