const int pwmPin = 4;  // PWM sinyalinin geldiği pin (Orange Cube'dan)
const int relayPin = 7; // Röle kontrol pini

void setup() {
  // Röle pinini çıkış olarak ayarlıyoruz
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW); // Başlangıçta röleyi kapalı tutuyoruz

  // PWM sinyalini okuyacağımız pin için giriş ayarı
  pinMode(pwmPin, INPUT);
  Serial.begin(9600);  // Seri haberleşmeyi başlat
}

void loop() {
  // PWM sinyalinin yüksek (HIGH) olduğu süreyi ölçüyoruz
  long pulseWidth = pulseIn(pwmPin, HIGH);

  // PWM sinyalinin süresi 1300 mikro saniyeden büyükse röleyi aç
  if (pulseWidth > 1300) {
    digitalWrite(relayPin, HIGH);  // Röleyi aç
    Serial.println("Röle Açık");
  } else {
    digitalWrite(relayPin, LOW);   // Röleyi kapat
    Serial.println("Röle Kapalı");
  }

  // Seri monitörde PWM süresini görüntüle
  Serial.print("PWM Süresi: ");
  Serial.println(pulseWidth);

  delay(100);  // Küçük bir gecikme ekleyerek sürekli okumayı sağlayalım
}