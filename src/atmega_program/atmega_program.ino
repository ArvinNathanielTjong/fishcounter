#include <ArduinoJson.h> // Library untuk mem-parsing dan membuat JSON
#include <avr/wdt.h>

// =================================================================
// KONSTANTA UNTUK PENGATURAN (Sama seperti sebelumnya)
// =================================================================
#define BATTERY_PIN A3 
#define ACOK_PIN A1
#define CHOK_PIN A0
#define SBC_EN_PIN 2
#define DISPLAY_EN_PIN 4
#define MOTOR1_PIN 5 // PD5
#define MOTOR2_PIN 6 // PD6
#define MOTOR3_PIN 9 // PB1
#define LED_PIN 3
#define BOOST_EN_PIN 7
#define CHARGE_EN_PIN 10

#define R1_DIVIDER 150000.0f //tanya pak yusuf
#define R2_DIVIDER 13000.0f //tanya pak yusuf
#define VOLTAGE_MAX 8.3f  // tanya pak yusuf
#define VOLTAGE_MIN 6.6f   // tanya pak yusuf
#define ADC_REFERENCE_VOLTAGE 5.0f 
#define NUM_SAMPLES 10 

int currentLedPwm = 0;
int currentMotorPwm = 0;
// =================================================================
// STRUKTUR DATA (Sama seperti sebelumnya)
// =================================================================
struct BatteryData {
  float voltage;
  int percentage;
};

// =================================================================
// PROGRAM UTAMA
// =================================================================

void setup() {
  Serial.begin(9600); 
  pinMode(ACOK_PIN, INPUT_PULLUP);
  pinMode(CHOK_PIN, INPUT_PULLUP);

  // ### PERUBAHAN: Logika untuk menyalakan SBC ###

  // 1. Atur pin SBC_EN sebagai OUTPUT
  pinMode(SBC_EN_PIN, OUTPUT);

    pinMode(MOTOR1_PIN, OUTPUT);
    pinMode(MOTOR2_PIN, OUTPUT);
    pinMode(MOTOR3_PIN, OUTPUT);

    pinMode(LED_PIN, OUTPUT);
    pinMode(BOOST_EN_PIN, OUTPUT);
    pinMode(CHARGE_EN_PIN, OUTPUT);


    // pastikan boost dan charge mati saat startup
  digitalWrite(BOOST_EN_PIN, LOW);
  digitalWrite(CHARGE_EN_PIN, LOW);

  // Pastikan SBC mati pada awalnya (opsional, tapi aman)
  digitalWrite(SBC_EN_PIN, LOW);
  digitalWrite(DISPLAY_EN_PIN, LOW); 

  // 2. Tunggu 1 detik setelah Arduino menyala
  delay(1000);

  // 3. Nyalakan power supply untuk SBC dengan mengirim sinyal HIGH
  digitalWrite(SBC_EN_PIN, HIGH);

  delay (3000);
  digitalWrite(DISPLAY_EN_PIN, HIGH);
}

// Fungsi baru untuk mengatur kecerahan LED
void setLedBrightness(int pwmValue) {
  currentLedPwm = constrain(pwmValue, 0, 255);
  analogWrite(LED_PIN, currentLedPwm);
}


// Fungsi baru untuk mengatur kecepatan semua motor
void setAllMotorsSpeed(int pwmValue) {
  // Pastikan nilai PWM berada dalam rentang yang valid (0-255)
  currentMotorPwm = constrain(pwmValue, 0, 255);
  
  analogWrite(MOTOR1_PIN, currentMotorPwm);
  analogWrite(MOTOR2_PIN, currentMotorPwm);
  analogWrite(MOTOR3_PIN, currentMotorPwm);
}


// ### Fungsi untuk mematikan semua sistem ###
void emergencyShutdown() {
  // 1. Matikan semua output utama (Motor, Display, SBC)
  setAllMotorsSpeed(0);
  setLedBrightness(0); // Matikan LED utama juga

  // kirim shutdown ke sbc (belom di implement)

  
  digitalWrite(DISPLAY_EN_PIN, LOW);
  digitalWrite(SBC_EN_PIN, LOW);

  // 2. Masuk ke loop hemat daya, tunggu sampai adaptor AC dicolokkan
  //    Loop ini akan terus berjalan selama pin ACOK masih HIGH (adaptor tidak terpasang)
  while (1) {
    // Kedipkan LED utama sebagai sinyal "Baterai Lemah, colok charger"
    analogWrite(LED_PIN, 80);
    delay(200);
    analogWrite(LED_PIN, 0);
    delay(1800); // Berkedip pelan untuk hemat daya
    //
    //delay nya ganti dengan sleep..wakeup pakai interrupt dari watchdog.
    //

    if (digitalRead(ACOK_PIN) == LOW)
         break;
  }

  // 3. Jika loop ini selesai, artinya adaptor AC SUDAH DICOLOK.
  //    Saatnya untuk memulai ulang sistem.
  
  // Nyalakan LED utama secara solid sebagai tanda "Akan restart..."
  analogWrite(LED_PIN, 255);
  delay(1000);

  // 4. Lakukan software reset menggunakan Watchdog Timer.
  //    Ini sama seperti menekan tombol reset fisik.
  wdt_enable(WDTO_15MS);
  while (true) {
    // Loop tak terbatas ini akan menyebabkan watchdog timer habis
    // dan memaksa mikrokontroler untuk reset.
  }
}


void loop() {
    
    //Prioritas utama, cek kondisi baterai setiap saat ###
  // Cek hanya jika adaptor TIDAK terpasang
  if (digitalRead(ACOK_PIN) == HIGH) {
    if (getBatteryInfo().percentage < 5) {
      emergencyShutdown();
    }
  }


  // Cek apakah ada data yang masuk dari port serial (dari Orange Pi)
  if (Serial.available() > 0) {
    // Baca string JSON yang masuk sampai karakter newline
    String jsonString = Serial.readStringUntil('\n');

    // Buat dokumen JSON untuk mem-parsing data masuk
    JsonDocument doc;
    deserializeJson(doc, jsonString);

    // Ambil perintah dari JSON
    const char* cmd = doc["cmd"];

    // Cek perintah "START MOTOR"
    if (cmd && strcmp(cmd, "START MOTOR") == 0) {
      // Saat start, atur ke kecepatan default (misal: 150 dari 255)
      setAllMotorsSpeed(150); 
    }
    // Cek perintah "STOP MOTOR"
    else if (cmd && strcmp(cmd, "STOP MOTOR") == 0) {
      // Saat stop, matikan motor
      setAllMotorsSpeed(0);
    }
    // Cek perintah baru "SET_SPEED"
    else if (cmd && strcmp(cmd, "SET_SPEED") == 0) {
      // Ambil nilai level kecepatan dari JSON (misal: 1-5)
      int speedLevel = doc["level"] | 0; // Default ke 0 jika tidak ada
      
      // Konversi level (1-5) menjadi nilai PWM (0-255)
      // Level 1 -> PWM ~51, Level 5 -> PWM 255
      int pwm = map(speedLevel, 1, 5, 51, 255);
      
      setAllMotorsSpeed(pwm);
    }

    else if (cmd && strcmp(cmd, "SET_BRIGHTNESS") == 0) {
      int brightnessLevel = doc["level"] | 0;
      int pwm = map(brightnessLevel, 1, 5, 51, 255);
      setLedBrightness(pwm);
    }

 // ###  blok untuk memulai proses charging ###
    else if (cmd && strcmp(cmd, "START_CHARGE") == 0) {
      // Hanya mulai charge jika adaptor AC terpasang
      if (digitalRead(ACOK_PIN) == LOW) {
        digitalWrite(BOOST_EN_PIN, HIGH);
        delay(3000); // Tunggu 3 detik
        digitalWrite(CHARGE_EN_PIN, HIGH);
      }
    }
    // ###  blok untuk menghentikan proses charging ###
    else if (cmd && strcmp(cmd, "STOP_CHARGE") == 0) {
      digitalWrite(CHARGE_EN_PIN, LOW);
      digitalWrite(BOOST_EN_PIN, LOW);
    }


    // for battery : 
    else if (cmd && strcmp(cmd, "BAT") == 0) {
      
      // 1. Panggil fungsi untuk mendapatkan data baterai
      BatteryData battery = getBatteryInfo();

      // 2. Buat dokumen JSON baru untuk respons
      JsonDocument responseDoc;
      responseDoc["type"] = "response";
      responseDoc["source"] = "BAT";
      responseDoc["voltage"] = battery.voltage;
      responseDoc["percentage"] = battery.percentage;

      // 3. Kirim respons JSON kembali ke Orange Pi
      serializeJson(responseDoc, Serial);
      Serial.println(); // Kirim newline sebagai penanda akhir pesan
    }
    // ### UNTUK AC ON OR OFF & CHARGING STATUS ###
    else if (cmd && strcmp(cmd, "GET_STATUS") == 0) {
      // Logika untuk mengecek status AC dan Charging
      JsonDocument responseDoc;
      responseDoc["type"] = "response";
      responseDoc["source"] = "STATUS";

      // Baca pin ACOK. LOW berarti AC terpasang (OK)
      bool ac_is_ok = (digitalRead(ACOK_PIN) == LOW);
      responseDoc["ac_ok"] = ac_is_ok;

      // ### PERUBAHAN: Baca pin CHOK ###
      // LOW berarti sedang mengisi daya (Normal Charging)
      bool is_charging = (digitalRead(CHOK_PIN) == LOW);
      responseDoc["is_charging"] = is_charging;

      serializeJson(responseDoc, Serial);
      Serial.println();
    }
  }
}

// =================================================================
// FUNGSI INTI
// =================================================================
BatteryData getBatteryInfo() {
  int totalAdcValue = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    totalAdcValue += analogRead(BATTERY_PIN);
    delay(1);
  }
  
  float averageAdcValue = totalAdcValue / (float)NUM_SAMPLES;
  float pinVoltage = (averageAdcValue / 1023.0f) * ADC_REFERENCE_VOLTAGE;
  float realVoltage = pinVoltage * (R1_DIVIDER + R2_DIVIDER) / R2_DIVIDER;
  float percentage = ((realVoltage - VOLTAGE_MIN) / (VOLTAGE_MAX - VOLTAGE_MIN)) * 100.0f;
  int constrainedPercentage = constrain(percentage, 0, 100);

  BatteryData data;
  data.voltage = realVoltage;
  data.percentage = constrainedPercentage;
  
  return data;
}

