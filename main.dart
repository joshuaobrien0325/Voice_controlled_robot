import 'dart:async';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:permission_handler/permission_handler.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) =>
      const MaterialApp(home: RecorderPage());
}

class RecorderPage extends StatefulWidget {
  const RecorderPage({super.key});
  @override
  State<RecorderPage> createState() => _RecorderPageState();
}

class _RecorderPageState extends State<RecorderPage> {
  final AudioRecorder _recorder = AudioRecorder();
  WebSocketChannel? _channel;
  StreamSubscription? _audioSubscription;
  bool _isRecording = false;
  String _status = 'Hold to record';
  
  // Store the single character command instead of a long transcript
  String _currentCommand = '-'; 

  // webSocket
  void _connectWebSocket() {
    _channel = WebSocketChannel.connect(
      Uri.parse('ws://127.0.0.1:8765'), 
    );

    _channel!.stream.listen(
      (message) {
        String charCommand = message.toString().toUpperCase();
        String fullWord = charCommand; // Default to the character itself

        // Translate the character into the full display word
        switch (charCommand) {
          case 'F': fullWord = 'FORWARD'; break;
          case 'B': fullWord = 'BACKWARD'; break;
          case 'S': fullWord = 'STOP'; break;
          case 'G': fullWord = 'GO'; break;
          case 'L': fullWord = 'LEFT'; break;
          case 'R': fullWord = 'RIGHT'; break;
          case '0': fullWord = 'ZERO'; break;
          case '1': fullWord = 'ONE'; break;
        }

        setState(() {
          _currentCommand = fullWord; 
          _status = 'Listening...';
        });
        
        
      },
      onError: (e) => setState(() => _status = 'WebSocket error: $e'),
      onDone: () => setState(() => _status = 'Connection closed'),
    );
  }

  // Permissions
  Future<bool> _requestPermission() async {
    final status = await Permission.microphone.request();
    return status.isGranted;
  }

  // Start Streaming 
  Future<void> _startRecording() async {
    if (!await _requestPermission()) {
      setState(() => _status = 'Microphone permission denied');
      return;
    }

    _connectWebSocket();

    final audioStream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,  
        sampleRate: 16000,                
        numChannels: 1,                   
      ),
    );

    _audioSubscription = audioStream.listen(
      (chunk) {
        if (_channel != null) {
          _channel!.sink.add(chunk);      
        }
      },
      onError: (e) => setState(() => _status = 'Stream error: $e'),
    );

    setState(() {
      _isRecording = true;
      _status = 'Recording...';
      _currentCommand = '-'; // Reset the command display
    });
  }

  // Stop Streaming
  Future<void> _stopRecording() async {
    await _recorder.stop();
    await _audioSubscription?.cancel();
    _audioSubscription = null;

    _channel?.sink.add('END_OF_STREAM');
    
    setState(() {
      _isRecording = false;
      _status = 'Processing...';
    });
  }

  @override
  void dispose() {
    _recorder.dispose();
    _audioSubscription?.cancel();
    _channel?.sink.close();
    super.dispose();
  }

 
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Voice Controller')),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            // Big Command Display
            Expanded(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      'Command Received:',
                      style: TextStyle(fontSize: 18, color: Colors.grey),
                    ),
                    const SizedBox(height: 16),
                    
                    
                    Container(
                      width: 300,  
                      height: 100, 
                      decoration: BoxDecoration(
                        color: Colors.grey[100],
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(color: Colors.blue.withValues(alpha: 0.3), width: 3),
                      ),
                      child: Center(
                        child: Text(
                          _currentCommand,
                          style: TextStyle(
                            fontSize: 36, 
                            fontWeight: FontWeight.bold,
                            color: _currentCommand == '-' ? Colors.grey[300] : Colors.blue[800],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),
            Text(_status, style: const TextStyle(fontSize: 16, color: Colors.grey)),
            const SizedBox(height: 16),

            // Hold-to-record button
            GestureDetector(
              onTapDown: (_) => _startRecording(),
              onTapUp: (_) => _stopRecording(),
              onTapCancel: () => _stopRecording(),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: _isRecording ? 110 : 90,
                height: _isRecording ? 110 : 90,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isRecording ? Colors.red : Colors.blue,
                  boxShadow: _isRecording
                      ? [BoxShadow(color: Colors.red.withValues(alpha: 0.4), blurRadius: 20, spreadRadius: 5)]
                      : [],
                ),
                child: Icon(
                  _isRecording ? Icons.mic : Icons.mic_none,
                  color: Colors.white,
                  size: 44,
                ),
              ),
            ),

            const SizedBox(height: 12),
            Text(
              _isRecording ? 'Release to stop' : 'Hold to record',
              style: const TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}