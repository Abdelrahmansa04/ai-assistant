const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" }
});

io.on('connection', (socket) => {
  console.log('🔌 A client connected:', socket.id);

  socket.on('disconnect', () => {
    console.log('❌ Client disconnected:', socket.id);
  });
});

// Custom endpoint for n8n to POST data and emit it via WebSocket
app.use(express.json());
app.post('/emit', (req, res) => {
  const data = req.body;
  console.log('📨 Data received from n8n:', data);

  // Emit to all connected clients
  io.emit('n8n-message', data);

  res.json({ status: 'emitted', data });
});

server.listen(5000, () => {
  console.log('🚀 WebSocket server listening on http://localhost:5000');
});