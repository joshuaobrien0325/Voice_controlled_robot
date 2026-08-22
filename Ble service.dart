import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_web_bluetooth/flutter_web_bluetooth.dart';
import 'package:flutter_web_bluetooth/js_web_bluetooth.dart';

// ── UUIDs matching the ESP32 firmware ──
const _serviceUuid = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
const _rxCharUuid  = '6e400002-b5a3-f393-e0a9-e50e24dcca9e';

enum EspConnectionState { disconnected, connecting, connected }

class BleService extends ChangeNotifier {
  EspConnectionState _state = EspConnectionState.disconnected;
  EspConnectionState get state => _state;

  String _status = 'Tap to connect';
  String get status => _status;

  BluetoothDevice? _device;
  BluetoothCharacteristic? _rxChar;

  bool get isSupported => FlutterWebBluetooth.instance.isBluetoothApiSupported;

  /// Open the browser's BLE device picker, connect, and discover the UART service
  Future<void> connect() async {
    if (_state != EspConnectionState.disconnected) return;

    if (!isSupported) {
      _setState(EspConnectionState.disconnected,
          'Web Bluetooth not supported — use Chrome');
      return;
    }

    _setState(EspConnectionState.connecting, 'Requesting device…');

    try {
      //  Show browser picker filtered to the UART service UUID
      final requestOptions = RequestOptionsBuilder(
        [RequestFilterBuilder(services: [_serviceUuid])],
      );
      final device =
          await FlutterWebBluetooth.instance.requestDevice(requestOptions);
      _device = device;

      // Connect GATT
      _setState(EspConnectionState.connecting, 'Connecting…');
      await device.connect();

      //  Discover UART service
      _setState(EspConnectionState.connecting, 'Discovering services…');
      final services = await device.discoverServices();
      final uartService = services.firstWhere(
        (s) => s.uuid == _serviceUuid,
        orElse: () => throw Exception('UART service not found'),
      );

      // 4. Get the RX characteristic 
      _rxChar = await uartService.getCharacteristic(_rxCharUuid);

      _setState(EspConnectionState.connected, 'Connected to ESP32');

      //  Listen for unexpected disconnects
      device.connected.listen((isConnected) {
        if (!isConnected) {
          _rxChar = null;
          _setState(
              EspConnectionState.disconnected, 'Disconnected — tap to reconnect');
        }
      });
    } on UserCancelledDialogError {
      _setState(EspConnectionState.disconnected, 'Pairing cancelled');
    } catch (e) {
      debugPrint('BLE connect error: $e');
      _setState(
          EspConnectionState.disconnected, 'Connection failed: ${e.toString()}');
    }
  }

  /// Write a single-character command 
  Future<void> sendCommand(String cmd) async {
    if (_rxChar == null || _state != EspConnectionState.connected) return;
    try {
      final bytes = Uint8List.fromList(cmd.codeUnits);
      await _rxChar!.writeValueWithoutResponse(bytes);
      debugPrint('BLE ➜ $cmd');
    } catch (e) {
      debugPrint('BLE write error: $e');
      _setState(EspConnectionState.disconnected, 'Write failed — reconnect');
    }
  }

  /// Disconnect 
  void disconnect() {
    try {
      sendCommand('S');
      _device?.disconnect();
    } catch (_) {}
    _rxChar = null;
    _device = null;
    _setState(EspConnectionState.disconnected, 'Disconnected');
  }

  //  internal 

  void _setState(EspConnectionState s, String msg) {
    _state = s;
    _status = msg;
    notifyListeners();
  }
}