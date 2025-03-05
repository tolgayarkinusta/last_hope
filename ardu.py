#include <Stepper.h>

#define STEPS_PER_REVOLUTION 200  // Stepper motorun tam dönüş için adım sayısı

Stepper myStepper(STEPS_PER_REVOLUTION, 8, 10, 9, 11);  // Pinler: (A, B, C, D)

int motorSpeed = 60;  // Motor hızı (RPM)
int pwmValue = 0;     // PWM sinyali için değişken

unsigned long pwmStartTime = 0;  // PWM sinyalinin aralıktaki kalmaya başladığı zamanı saklar
bool pwmTimerStarted = false;    // Zamanlayıcının başlatılıp başlatılmadığını kontrol eder

void setup() {
  myStepper.setSpeed(motorSpeed);  // Motor hızını ayarla
  Serial.begin(9600);              // Seri haberleşmeyi başlat
  Serial.println("Stepper motor kontrolü başladı!");
}

void loop() {
  pwmValue = analogRead(A0);  // A0 pininden PWM sinyalini oku

  // PWM sinyali 1900-2000 aralığındaysa
  if (pwmValue >= 1900 && pwmValue <= 2000) {
    // Eğer zamanlayıcı daha önce başlatılmamışsa başlat
    if (!pwmTimerStarted) {
      pwmStartTime = millis();  // Şu anki zamanı kaydet
      pwmTimerStarted = true;
    }
    // Eğer sinyal bu aralıkta 1 saniyeden uzun sürüyorsa motoru döndür
    else if (millis() - pwmStartTime >= 1000) {
      Serial.println("PWM sinyali 1900-2000 aralığında 1 saniyedir, motor 67 adım döndürülüyor...");
      myStepper.step(67);  // 67 adım döndür
      Serial.println("Motor 67 adım döndü.");
      pwmTimerStarted = false;  // Zamanlayıcıyı sıfırla
    }
  }
  else {
    // PWM sinyali aralıkta değilse zamanlayıcıyı sıfırla
    pwmTimerStarted = false;
  }
}







