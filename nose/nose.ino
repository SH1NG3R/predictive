#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <SPI.h>
#include <SD.h>
#include <Adafruit_SSD1306.h>
#include <Firebase_ESP_Client.h>
#include <Adafruit_MLX90640.h>
#include "time.h"
#include <PCF8563.h>

const int chipSelect = D2;  // SS pin for SD card
const uint8_t RTC_ADDRESS = 0x51;  // Dirección del RTC PCF8563

// Configuración WiFi
const char* ssid = "CCL";
const char* password = "contrasena";

// Configuración Firebase
#define API_KEY "aaaa"
#define DATABASE_URL "bbbbbbbbb"

// Configuración OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Objetos
Adafruit_ADS1115 ads;
Adafruit_MLX90640 mlx;
float frame[32*24];  // Buffer para frame térmico
PCF8563 pcf;         // Objeto RTC PCF8563
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;


// Pines
const int BUTTON_PIN = D1;
const int VIBRATION_PIN = A0;

// Variables globales
String currentFileName;
bool isLogging = false;
const char* ntpServer = "cl.pool.ntp.org";
const long gmtOffset_sec = -14400;
const int daylightOffset_sec = 0;


void setup() {
    Serial.begin(115200);
    Wire.begin();
    Wire.setClock(400000); // Configurar I2C a 400KHz
    
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    
    // Inicializar OLED
    if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println(F("SSD1306 allocation failed"));
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    
    // Conectar WiFi
    setupWiFi();
    
    // Configurar tiempo
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    
    // Inicializar MLX90640
    if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
        showError("MLX90640 failed");
    }
    
    mlx.setMode(MLX90640_CHESS);
    mlx.setResolution(MLX90640_ADC_18BIT);
    mlx.setRefreshRate(MLX90640_2_HZ);
    
    // Inicializar ADS1115
    if (!ads.begin(0x48)) {
        if (!ads.begin(0x4A)) {
            showError("ADS1115 failed");
        }
    }
    
    // Inicializar SD
    if (!SD.begin(chipSelect)) {
        showError("SD failed");
    }
    
    // Inicializar RTC PCF8563
    pcf.init();
    
    // Configurar Firebase
    config.api_key = API_KEY;
    config.database_url = DATABASE_URL;
    Firebase.begin(&config, &auth);
    
    display.clearDisplay();
    display.setCursor(0,0);
    display.println("System Ready!");
    display.display();
}
void loop() {
  if (digitalRead(BUTTON_PIN) == LOW && !isLogging) {
    isLogging = true;
    createNewFile();
  }
  
  if (isLogging) {
    logData();
    delay(1000); // Ajustar según necesidades
  }
}

void setupWiFi() {
  WiFi.begin(ssid, password);
  display.clearDisplay();
  display.setCursor(0,0);
  display.println("Connecting WiFi...");
  display.display();
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  
  display.println("Connected!");
  display.display();
}

void createNewFile() {
  struct tm timeinfo;
  if(!getLocalTime(&timeinfo)){
    showError("Failed to obtain time");
    return;
  }
  
  char fileName[32];
  strftime(fileName, sizeof(fileName), "/data_%Y_%m_%d.csv", &timeinfo);
  currentFileName = String(fileName);
  
  if (!SD.exists(currentFileName)) {
    File dataFile = SD.open(currentFileName, FILE_WRITE);
    dataFile.println("Fecha,Hora,Vibracion,Corriente1,Corriente2,Corriente3,TempMin,TempMax");
    dataFile.close();
  }
}

void logData() {
  struct tm timeinfo;
  if(!getLocalTime(&timeinfo)){
    return;
  }
  
  // Leer sensores de corriente y vibración
  float vibration = analogRead(VIBRATION_PIN);
  float current1 = ads.computeVolts(ads.readADC_SingleEnded(0)) * 6;
  float current2 = ads.computeVolts(ads.readADC_SingleEnded(1)) * 6;
  float current3 = ads.computeVolts(ads.readADC_SingleEnded(2)) * 6;
  
  // Leer sensor térmico
  float temp_min = 100.0;  // Inicializar con valor alto
  float temp_max = -40.0;  // Inicializar con valor bajo
  
  if (mlx.getFrame(frame) != 0) {
    Serial.println("Failed to get frame");
  } else {
    // Encontrar temperaturas máxima y mínima
    for (int i = 0; i < 32 * 24; i++) {
      if (frame[i] > temp_max) temp_max = frame[i];
      if (frame[i] < temp_min) temp_min = frame[i];
    }
  }
  
  // Formatear datos
  char dateStr[11];
  char timeStr[9];
  strftime(dateStr, sizeof(dateStr), "%Y-%m-%d", &timeinfo);
  strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &timeinfo);
  
  // Crear string de datos
  String dataString = String(dateStr) + "," + String(timeStr) + "," + 
                     String(vibration) + "," + String(current1) + "," + 
                     String(current2) + "," + String(current3) + "," +
                     String(temp_min) + "," + String(temp_max);
  
  // Guardar en SD
  File dataFile = SD.open(currentFileName, FILE_APPEND);
  if (dataFile) {
    dataFile.println(dataString);
    dataFile.close();
  }
  
  // Enviar a Firebase
  String path = "sensor_data/" + String(dateStr);
  Firebase.RTDB.pushString(&fbdo, path.c_str(), dataString.c_str());
  
  // Actualizar display
  display.clearDisplay();
  display.setCursor(0,0);
  display.println("Logging...");
  display.println(timeStr);
  display.println("Vib: " + String(vibration));
  display.println("I1: " + String(current1));
  display.println("Tmin: " + String(temp_min,1));
  display.println("Tmax: " + String(temp_max,1));
  display.display();
}

void showError(const char* error) {
  display.clearDisplay();
  display.setCursor(0,0);
  display.println("ERROR:");
  display.println(error);
  display.display();
  while(1) delay(100);
}
