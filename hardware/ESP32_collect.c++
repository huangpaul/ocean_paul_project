#include <Wire.h>
#include "RTClib.h"
#include "DHT.h"
#include <SPI.h>
#include <SD.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <HTTPClient.h>
#include "time.h"

// 💡 引入剛剛建立的隱私密鑰檔
#include "secrets.h"

WiFiMulti wifiMulti;

// 💡 改為讀取 secrets.h 的變數，避免 IP 裸奔
const char* serverUrl = SECRET_SERVER_URL;

const char* ntpServer = "pool.ntp.org";      
const long  gmtOffset_sec = 28800; // GMT+8
const int   daylightOffset_sec = 0;          

unsigned long lastWifiCheckTime = 0;
const unsigned long wifiCheckInterval = 60000; 

// 非阻塞定時器
unsigned long lastLogTime = 0;
const unsigned long logInterval = 30000; // 嚴格 30 秒

RTC_DS3231 rtc;
#define I2C_SDA 8
#define I2C_SCL 9

#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

#define SD_SCK  12
#define SD_MISO 13
#define SD_MOSI 11
#define SD_CS   10

#define PH_PIN  1  

// 🔥 續傳專用核心狀態變數
bool isUploadingQueue = false;
unsigned long queueFilePosition = 0; // 記憶上一次傳到檔案的哪一個字元位置

void setup() {
  Serial.begin(115200);
  delay(2000); 

  Serial.println("=============================================");
  Serial.println("【無漏洞斷點續傳版】硬體自動續傳雙軌系統 (安全重構版)");
  Serial.println("=============================================");

  Wire.begin(I2C_SDA, I2C_SCL);
  if (!rtc.begin(&Wire)) Serial.println("【⚠️ 警告】找不到 RTC 時間模組！");

  dht.begin();

  Serial.println("正在掛載右側 5V 獨立 SD 卡...");
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS)) {
    Serial.println("【❌ 致命錯誤】SD 卡掛載失敗！");
    while (1) delay(10); 
  }
  Serial.println("【確定】SD 卡掛載成功！");

  WiFi.mode(WIFI_STA);
  // 💡 改為安全讀取隱私設定
  wifiMulti.addAP(WIFI_SSID_1, WIFI_PASS_1); 
  wifiMulti.addAP(WIFI_SSID_2, WIFI_PASS_2);            
  wifiMulti.addAP(WIFI_SSID_3, WIFI_PASS_3);             
  
  Serial.println("[系統] 開始背景搜尋 Wi-Fi...");
  wifiMulti.run(); 

  analogReadResolution(12);

  if (!SD.exists("/subsea_log.csv")) {
    File dataFile = SD.open("/subsea_log.csv", FILE_WRITE);
    if (dataFile) {
      dataFile.println("Date,Time,Cabin_Temp_C,Cabin_Hum_%,pH_Raw,pH_mV,Wifi_Status");
      dataFile.close();
    }
  }

  Serial.println("\n>>> 全系統就緒！定時與智慧續傳就位 <<<");
}

void loop() {
  unsigned long currentMillis = millis();
  uint8_t wifiStatus = wifiMulti.run(); 
  bool isConnected = (WiFi.status() == WL_CONNECTED);
  String wifi_status_str = isConnected ? "CONNECTED" : "OFFLINE";

  // --- 【1. 背景網路定時對時】 ---
  if (isConnected && (currentMillis - lastWifiCheckTime >= wifiCheckInterval || lastWifiCheckTime == 0)) {
    lastWifiCheckTime = currentMillis;
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    struct tm timeinfo;
    if (getLocalTime(&timeinfo)) {
      rtc.adjust(DateTime(timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday, 
                          timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec));
      Serial.printf("\n[🎉 背景對時] 時間已同步。當前網路: %s\n", WiFi.SSID().c_str());
    }
  }

  // --- 【2. 嚴格 30 秒非阻塞採樣與智能分流】 ---
  if (currentMillis - lastLogTime >= logInterval || lastLogTime == 0) {
    lastLogTime = currentMillis; 
    
    DateTime now = rtc.now();
    float cabin_hum = dht.readHumidity();
    float cabin_temp = dht.readTemperature();
    int ph_raw = analogRead(PH_PIN);
    float ph_mv = ph_raw * (3300.0 / 4095.0);

    if (!isnan(cabin_hum) && !isnan(cabin_temp)) {
      char timeStr[25];
      sprintf(timeStr, "%04d-%02d-%02d %02d:%02d:%02d", now.year(), now.month(), now.day(), now.hour(), now.minute(), now.second());

      char dataString[180];
      sprintf(dataString, "%04d-%02d-%02d,%02d:%02d:%02d,%.1f,%.1f,%d,%.0f,%s", 
              now.year(), now.month(), now.day(), now.hour(), now.minute(), now.second(), 
              cabin_temp, cabin_hum, ph_raw, ph_mv, wifi_status_str.c_str());

      if (isConnected && !isUploadingQueue) { 
        // 💡 只有在「背景沒有在抓歷史檔案」時，才允許即時直發，避免通道撞車
        HTTPClient http;
        http.begin(serverUrl);
        http.addHeader("Content-Type", "application/json");

        String jsonPayload = "{\"time\":\"" + String(timeStr) + 
                             "\",\"temp\":" + String(cabin_temp) + 
                             ",\"humi\":" + String(cabin_hum) + 
                             ",\"ph_raw\":" + String(ph_raw) + 
                             ",\"ph_mv\":" + String((int)ph_mv) + "}";
        
        int httpResponseCode = http.POST(jsonPayload);
        http.end();

        if (httpResponseCode > 0) {
          Serial.printf("🌐【即時發送】最新數據直達伺服器！狀態碼: %d\n", httpResponseCode);
        } else {
          Serial.printf("⚠️【即時發送失敗】轉入 SD 卡暫存。錯誤: %d\n", httpResponseCode);
          File dataFile = SD.open("/subsea_log.csv", FILE_APPEND);
          if (dataFile) { dataFile.println(dataString); dataFile.close(); }
        }
      } 
      else {
        // 離線狀態，或者背景正在全力續傳時，新數據一律先寫入 SD 卡排隊，絕對不插隊插壞 API
        File dataFile = SD.open("/subsea_log.csv", FILE_APPEND);
        if (dataFile) {
          dataFile.println(dataString);
          dataFile.close();
          Serial.print("💾【分流儲存】已寫入 SD 卡 ➜ "); 
          Serial.println(dataString);
        }
      }
    }
  }

  // --- --- 【3. 背景空閒續傳機制（⚙️ 中斷點記憶版）】 --- ---
  if (isConnected) {
    // 檢查點 A：如果當前沒有佇列檔，且原本的日誌檔有離線數據，才進行「瞬間更名搬移」
    if (!SD.exists("/queue.csv") && SD.exists("/subsea_log.csv")) {
      File checkFile = SD.open("/subsea_log.csv", FILE_READ);
      size_t fileSize = checkFile.size();
      checkFile.close();

      if (fileSize > 75) {
        Serial.println("\n[🔄 續傳啟動] 偵測到離線數據，搬移至獨立佇列檔...");
        SD.rename("/subsea_log.csv", "/queue.csv");
        queueFilePosition = 0; // 新佇列，從頭讀取

        // 重新建立空的日誌檔
        File newLog = SD.open("/subsea_log.csv", FILE_WRITE);
        if (newLog) {
          newLog.println("Date,Time,Cabin_Temp_C,Cabin_Hum_%,pH_Raw,pH_mV,Wifi_Status");
          newLog.close();
        }
      }
    }

    // 檢查點 B：如果佇列檔存在，利用 loop 空閒時間斷點續傳
    if (SD.exists("/queue.csv") && !isUploadingQueue) {
      isUploadingQueue = true;

      File uploadFile = SD.open("/queue.csv", FILE_READ);
      if (uploadFile) {
        // 💡 核心修正：跳躍到上一次被強行中斷的檔案字元位置
        if (queueFilePosition > 0) {
          uploadFile.seek(queueFilePosition);
        } else {
          if (uploadFile.available()) uploadFile.readStringUntil('\n'); // 第一次跳過標題
        }
        
        int successCount = 0;
        while (uploadFile.available() && WiFi.status() == WL_CONNECTED) {
          
          // 🛡️ 嚴格時間保全：如果距離下一次 30 秒採樣點只剩 2.5 秒，立刻保存指針，退出！
          if (millis() - lastLogTime > (logInterval - 2500)) {
            queueFilePosition = uploadFile.position(); // 🔥 記憶目前讀到哪裡
            Serial.printf("[⚡ 續傳暫停] 留空檔給定時採樣。目前已進度補傳 %d 筆，位置記在 %d\n", successCount, queueFilePosition);
            break; 
          }

          String line = uploadFile.readStringUntil('\n');
          line.trim();
          if (line.length() == 0) continue;

          int c1 = line.indexOf(','); 
          int c2 = line.indexOf(',', c1+1);
          int c3 = line.indexOf(',', c2+1); 
          int c4 = line.indexOf(',', c3+1);
          int c5 = line.indexOf(',', c4+1); 
          int c6 = line.indexOf(',', c5+1);

          if(c1 <= 0 || c2 <= 0 || c3 <= 0 || c4 <= 0 || c5 <= 0) continue; 

          String logDate   = line.substring(0, c1);
          String logTime   = line.substring(c1+1, c2);
          String logTemp   = line.substring(c2+1, c3);
          String logHumi   = line.substring(c3+1, c4);
          String logPhRaw  = line.substring(c4+1, c5);
          String logPhMv   = line.substring(c5+1, c6);

          HTTPClient http;
          http.begin(serverUrl);
          http.addHeader("Content-Type", "application/json");

          String jsonPayload = "{\"time\":\"" + logDate + " " + logTime + "\",\"temp\":" + logTemp + ",\"humi\":" + logHumi + ",\"ph_raw\":" + logPhRaw + ",\"ph_mv\":" + logPhMv + "}";
          int httpResponseCode = http.POST(jsonPayload);
          http.end();

          if (httpResponseCode > 0) {
            successCount++;
          }
          yield(); 
          delay(15); 
        }
        
        // 檢查是不是真的全檔案都讀完了
        bool fullyDone = !uploadFile.available();
        uploadFile.close();
        
        if (fullyDone) {
          SD.remove("/queue.csv"); // 只有完完全全傳完，才把歷史隊列刪掉！
          queueFilePosition = 0;
          Serial.printf("[🎉 續傳完成] 佇列中所有歷史數據已全部補傳完畢！共計 %d 筆。\n\n", successCount);
        }
      }
      isUploadingQueue = false;
    }
  }

  delay(1); 
}