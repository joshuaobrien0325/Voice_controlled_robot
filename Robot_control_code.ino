#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

//BLE UUIDs 
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

// Motor A Pins (Left Motor)
const int AIN1 = 2;   
const int AIN2 = 4;   
const int PWMA = 1;   

//Motor B Pins (Right Motor)
const int BIN1 = 13; 
const int BIN2 = 14;  
const int PWMB = 15;  

// Sensor Pins
const int trigPin = 7;
const int echoPin = 8;
const float STOP_DISTANCE_CM = 15.0;

// stes default state as stopped and default speed as medium speed 3
char currentCommand = 'S'; 
int currentSpeed = 150; 

// This intterupts whenever a command is recieved 
class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
      String rxValue = pCharacteristic->getValue();

      if (rxValue.length() > 0) {
        char receivedChar = rxValue[0]; // Look at the first character sent
        Serial.print("Received BLE Command: ");
        Serial.println(receivedChar);

        // control speed based off recieved number 
        if (receivedChar == '1') currentSpeed = 100;
        else if (receivedChar == '2') currentSpeed = 140;
        else if (receivedChar == '3') currentSpeed = 180;
        else if (receivedChar == '4') currentSpeed = 220;
        else if (receivedChar == '5') currentSpeed = 255;
        
        // Directional Commands
        else if (receivedChar == 'F' || receivedChar == 'B' || 
                 receivedChar == 'L' || receivedChar == 'R' || 
                 receivedChar == 'S') {
          currentCommand = receivedChar;
        }
      }
    }
};

void setup() {
  Serial.begin(115200);

  // Set Motor Pins as Outputs
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(PWMB, OUTPUT);

  // Sensor Setup
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // Ensure motors are off at boot
  stopMotors();

  // BLE Setup 
  Serial.println("Starting BLE Server...");
  BLEDevice::init("ESP32_ROBOT"); 
  BLEServer *pServer = BLEDevice::createServer();
  
  BLEService *pService = pServer->createService(SERVICE_UUID);
  
  BLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
                       CHARACTERISTIC_UUID_RX,
                       BLECharacteristic::PROPERTY_WRITE
                     );

  pRxCharacteristic->setCallbacks(new MyCallbacks());
  pService->start();
  
  // Start broadcasting
   BLEAdvertising *pAdvertising = pServer->getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);   // ← this was missing
  pAdvertising->setScanResponse(true);
  pAdvertising->start();
  Serial.println("BLE Ready. Waiting for connection. ");
}

void loop() {
  // 1. check distance sensor first
  float distance = getDistance();
  
  // 2. ensures stop only occurs when in forward movement
  if (currentCommand == 'F' && distance < STOP_DISTANCE_CM) {
    Serial.print("OBSTACLE AT ");
    Serial.print(distance);
    Serial.println(" cm! Emergency Stop.");
    currentCommand = 'S'; 
  }

  // 3. Execute the current command
  executeMovement();

  delay(20); 
}

// funbc to read ultrasonic sensor
float getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // 30ms timeout prevents the ESP32 from freezing
  long duration = pulseIn(echoPin, HIGH, 30000);

  if (duration == 0) return 999.0; // Safe distance if no echo
  return (duration * 0.0343) / 2;
}

// translateing the commands to actual movement
void executeMovement() {
  switch (currentCommand) {
    case 'F': // Forward
      digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
      digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
      analogWrite(PWMA, currentSpeed);
      analogWrite(PWMB, currentSpeed);
      break;
      
    case 'B': // Backward
      digitalWrite(AIN1, LOW); digitalWrite(AIN2, HIGH);
      digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);
      analogWrite(PWMA, currentSpeed);
      analogWrite(PWMB, currentSpeed);
      break;
      
    case 'L': // Turn Left 
      digitalWrite(AIN1, LOW); digitalWrite(AIN2, HIGH);
      digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
      analogWrite(PWMA, currentSpeed);
      analogWrite(PWMB, currentSpeed);
      break;
      
    case 'R': // Turn Right 
      digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
      digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);
      analogWrite(PWMA, currentSpeed);
      analogWrite(PWMB, currentSpeed);
      break;
      
    case 'S': // Stop
    default:
      stopMotors();
      break;
  }
}

void stopMotors() {
  digitalWrite(AIN1, LOW); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW); digitalWrite(BIN2, LOW);
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}