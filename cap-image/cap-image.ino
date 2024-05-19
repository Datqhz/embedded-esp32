#include <WiFi.h>
#include "esp_camera.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <Arduino.h>
#include <ArduinoJson.h>

#include <TensorFlowLite_ESP32.h>
#include "c_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

const char* ssid = "Q01";
const char* password = "0832367943";
String serverName = "192.168.1.15";      // Flask upload route

const int serverPort = 5000;

// const int timerInterval = 500;    // time (milliseconds) between each HTTP POST image
// unsigned long previousMillis = 0; 

boolean predictFlag = true;
boolean notifyFlag = false;
int relay = 2;

int person_idx = -1 ;

WiFiClient client;

// Define esp-32 pin
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_LED_PIN 4


namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 32 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace


//define task
void TaskPredict(void *pvParameters);
void TaskNotify(void *pvParameters);


void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); 
  Serial.begin(115200);

  // pinMode(relay, OUTPUT) // relay
  // digitalWrite(relay, LOW)

  WiFi.mode(WIFI_STA);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);  
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();
  Serial.print("ESP32-CAM IP Address: ");
  Serial.println(WiFi.localIP());

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // init with high specs to pre-allocate larger buffers
  if(psramFound()){
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 10;  //0-63 lower number means higher quality
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_CIF;
    config.jpeg_quality = 12;  //0-63 lower number means higher quality
    config.fb_count = 1;
  }
  
  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    delay(1000);
    ESP.restart();
  }


  // set up model for detect
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  // Map the model into a usable data structure
  model = tflite::GetModel(c_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  //setup RTOS scheduler
  xTaskCreatePinnedToCore (
    TaskPredict,     // Function to implement the task
    "Predict",   // Name of the task
    3000,      // Stack size in bytes
    NULL,      // Task input parameter
    0,         // Priority of the task
    NULL,      // Task handle.
    0          // Core where the task should run
  );

  xTaskCreatePinnedToCore (
    TaskNotify,     // Function to implement the task
    "Notify",   // Name of the task
    2000,      // Stack size in bytes
    NULL,      // Task input parameter
    0,         // Priority of the task
    NULL,      // Task handle.
    0          // Core where the task should run
  );
}

//send image for train to flask server
String sendPhoto() {
  String getAll;
  String getBody;

  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    ESP.restart();
  }
  
  Serial.println("Connecting to server: " + serverName);

  if (client.connect(serverName.c_str(), serverPort)) {
    Serial.println("Connection successful!");    
    String head = "--ESP32\r\nContent-Disposition: form-data; name=\"imageFile\"; filename=\"esp32-cam.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
    String tail = "\r\n--ESP32--\r\n";

    uint16_t imageLen = fb->len;
    uint16_t extraLen = head.length() + tail.length();
    uint16_t totalLen = imageLen + extraLen;
  
    client.println("POST /receive-image HTTP/1.1");
    client.println("Host: " + serverName);
    client.println("Content-Length: " + String(totalLen));
    client.println("Content-Type: multipart/form-data; boundary=ESP32");
    client.println();
    client.print(head);
  
    uint8_t *fbBuf = fb->buf;
    size_t fbLen = fb->len;
    for (size_t n=0; n<fbLen; n=n+1024) {
      if (n+1024 < fbLen) {
        client.write(fbBuf, 1024);
        fbBuf += 1024;
      }
      else if (fbLen%1024>0) {
        size_t remainder = fbLen%1024;
        client.write(fbBuf, remainder);
      }
    }   
    client.print(tail);
    
    esp_camera_fb_return(fb);
    
    int timoutTimer = 10000;
    long startTimer = millis();
    boolean state = false;
    
    while ((startTimer + timoutTimer) > millis()) {
      Serial.print(".");
      delay(100);      
      while (client.available()) {
        char c = client.read();
        if (c == '\n') {
          if (getAll.length()==0) { state=true; }
          getAll = "";
        }
        else if (c != '\r') { getAll += String(c); }
        if (state==true) { getBody += String(c); }
        startTimer = millis();
      }
      if (getBody.length()>0) { break; }
    }
    Serial.println();
    client.stop();
    Serial.println(getBody);
  }
  else {
    getBody = "Connection to " + serverName +  " failed.";
    Serial.println(getBody);
  }
  return getBody;
}

//send image to flask server and receive image after process (2d)
String predict2d(){
  String getAll;
  String getBody;

  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    ESP.restart();
  }
  
  Serial.println("Connecting to server: " + serverName);

  if (client.connect(serverName.c_str(), serverPort)) {
    Serial.println("Connection successful!");    
    String head = "--ESP32\r\nContent-Disposition: form-data; name=\"imageFile\"; filename=\"esp32-cam.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
    String tail = "\r\n--ESP32--\r\n";

    uint16_t imageLen = fb->len;
    uint16_t extraLen = head.length() + tail.length();
    uint16_t totalLen = imageLen + extraLen;
  
    client.println("POST /process-image HTTP/1.1");
    client.println("Host: " + serverName);
    client.println("Content-Length: " + String(totalLen));
    client.println("Content-Type: multipart/form-data; boundary=ESP32");
    client.println();
    client.print(head);
  
    uint8_t *fbBuf = fb->buf;
    size_t fbLen = fb->len;
    for (size_t n=0; n<fbLen; n=n+1024) {
      if (n+1024 < fbLen) {
        client.write(fbBuf, 1024);
        fbBuf += 1024;
      }
      else if (fbLen%1024>0) {
        size_t remainder = fbLen%1024;
        client.write(fbBuf, remainder);
      }
    }   
    client.print(tail);
    
    esp_camera_fb_return(fb);
    
    int timoutTimer = 10000;
    long startTimer = millis();
    boolean state = false;
    
    while ((startTimer + timoutTimer) > millis()) {
      Serial.print(".");
      delay(100);      
      while (client.available()) {
        char c = client.read();
        if (c == '\n') {
          if (getAll.length()==0) { state=true; }
          getAll = "";
        }
        else if (c != '\r') { getAll += String(c); }
        if (state==true) { getBody += String(c); }
        startTimer = millis();
      }
      if (getBody.length()>0) { break; }
    }
    Serial.println();
    client.stop();
    Serial.println(getBody);
    // Mở kết nối mới để tải xuống hình ảnh đã được cắt từ máy chủ
    if (client.connect(serverName.c_str(), serverPort)) {
      client.println("GET /image_2d HTTP/1.1");
      client.println("Host: " + serverName);
      client.println();

      // Đọc phản hồi từ máy chủ và lưu hình ảnh vào bộ nhớ đệm ESP32-CAM
      int httpStatus = -1;
      while (client.connected()) {
        String line = client.readStringUntil('\n');
        // get http status
        if (line.startsWith("HTTP/1.1")) {
          httpStatus = line.substring(9, 12).toInt();
        }
        if (line == "\r") {
          // headers are finished, time to get the response
          break;
        }
      }
      if (httpStatus == 404) {
        Serial.println("Image not found on server");
        return "error";
      } else if (httpStatus != 200) {
        Serial.print("Server returned non-OK status: ");
        Serial.println(httpStatus);
        return "error";
      }
      String response = "";
      while (client.available()) {
        response += client.readString();
      }
       // Parse JSON
       Serial.print("free heap:");
       Serial.println(ESP.getFreeHeap());
       const size_t capacity = JSON_ARRAY_SIZE(12) * 12 * JSON_ARRAY_SIZE(1);
      DynamicJsonDocument doc(capacity); // Sử dụng DynamicJsonDocument với kích thước phù hợp
      DeserializationError error = deserializeJson(doc, response);

      if (error) {
        Serial.print(F("deserializeJson() failed: "));
        Serial.println(error.f_str());
        return "error";
      }

      // Lấy dữ liệu từ JSON và gán vào mảng image_data
      float image_data[1][12][12][1];
      JsonArray data = doc.as<JsonArray>();
      for (int i = 0; i < 12; i++) {
        JsonArray row = data[i];
        for (int j = 0; j < 12; j++) {
          image_data[0][i][j][0] = row[j];
        }
      } 
      memcpy(model_input->data.f, image_data, sizeof(float) * 1 * 12 * 12 * 1);

      // // Run inference
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed on input: %f\n", model_input->data.f[0]);
        return "error";
      }

      // Lấy kết quả từ output tensor
      int max_prob = 0;
      for (int i = 0; i < model_output->bytes / sizeof(float); ++i) {
        Serial.print("output data[");
        Serial.print(i);
        Serial.print("] = ");
        Serial.println(model_output->data.f[i]);
        if(model_output->data.f[i]>model_output->data.f[max_prob]){
          max_prob = i;
        }
      }
      if(model_output->data.f[max_prob]>=0.5){
        Serial.print("predict: ");
        Serial.println(max_prob);
        person_idx = max_prob;
        predictFlag = false;
        notifyFlag = true;
      }else {
        Serial.print("unknow");
      }
      Serial.println();
    
    }else {
      return "error";
    }
  }
  else {
    getBody = "Connection to " + serverName +  " failed.";
    Serial.println(getBody);
  }
  return getBody;
}

void notify() {
  if (client.connect(serverName.c_str(), serverPort)) {
    Serial.println("Connection successful!");  
    // Tạo JSON
    StaticJsonDocument<200> doc;
    doc["idx"] = person_idx;

    // Serialize JSON thành chuỗi
    String jsonString;
    serializeJson(doc, jsonString);

    // Gửi yêu cầu POST
    client.print("POST /save-open-door HTTP/1.1\r\n");
    client.print("Host: ");
    client.print(serverName);
    client.print("\r\n");
    client.print("Content-Type: application/json\r\n");
    client.print("Content-Length: ");
    client.print(jsonString.length());
    client.print("\r\n\r\n");
    client.print(jsonString);

    Serial.println("Request sent to server.");

    // Đọc và in phản hồi từ server
    while (client.available()) {
      String line = client.readStringUntil('\r');
      Serial.print(line);
    }
  }else {
    Serial.println("Connection fail!");  
  }
}
// send image to flask server and receive image after process(1d)
String predict1d(){
  if (client.connect(serverName.c_str(), serverPort)) {
      client.println("GET /image_1d HTTP/1.1");
      client.println("Host: " + serverName);
      client.println();

      // Đọc phản hồi từ máy chủ và lưu hình ảnh vào bộ nhớ đệm ESP32-CAM
      while (client.connected()) {
        String line = client.readStringUntil('\n');
        if (line == "\r") {
          // headers are finished, time to get the response
          break;
        }
      }
      String response = "";
      while (client.available()) {
        response += client.readStringUntil('\n');
      }
      const size_t capacity = JSON_ARRAY_SIZE(144) + JSON_OBJECT_SIZE(2);
      DynamicJsonDocument doc(capacity);
      deserializeJson(doc, response);
      JsonArray array = doc.as<JsonArray>();
      // Sử dụng bộ nhớ heap thay vì stack
      float* img_1d = (float*)malloc(144 * sizeof(float));
      if (img_1d == nullptr) {
        Serial.println("Failed to allocate memory for img_1d");
        return "error";
      }

      for (int i = 0; i < 144; i++) {
        img_1d[i] = array[i];
        Serial.print("img_1d[");
        Serial.print(i);
        Serial.print("] = ");
        Serial.println(img_1d[i]);
      }

      float (*image_3d)[144][1] = (float(*)[144][1])malloc(sizeof(float) * 1 * 144 * 1);
      if (image_3d == nullptr) {
        Serial.println("Failed to allocate memory for image_3d");
        free(img_1d); // Giải phóng bộ nhớ img_1d trước khi trả về
        return "error";
      }

      for (int i = 0; i < 144; ++i) {
        (*image_3d)[i][0] = img_1d[i];
      }

      TfLiteTensor* input = interpreter->input(0);
      memcpy(input->data.f, image_3d, sizeof(float) * 1 * 144 * 1);
      // Print the data in the input tensor
      for (int i = 0; i < 144; ++i) {
        Serial.print("input data[");
        Serial.print(i);
        Serial.print("] = ");
        Serial.println(input->data.f[i]);
      }
      // Run inference
          // Run inference
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed on input: %f\n", input->data.f[0]);
        free(img_1d); // Giải phóng bộ nhớ img_1d trước khi trả về
        free(image_3d); // Giải phóng bộ nhớ image_3d trước khi trả về
        return "error";
      }
      TfLiteTensor* output = interpreter->output(0);
      for (int i = 0; i < output->bytes / sizeof(float); ++i) {
        Serial.print("output data[");
        Serial.print(i);
        Serial.print("] = ");
        Serial.println(output->data.f[i]);
      }
    free(img_1d); // Giải phóng bộ nhớ img_1d sau khi sử dụng
    free(image_3d);
    }else {
      return "error";
    }
}

void TaskPredict(void *pvParameters) {
    while (true) {
        if(predictFlag){
          predict2d();
          delay(500); // sau mỗi 0.5 giây thì task được lặp lại
        }else {
          delay(100);
        }
       
    }
}
void TaskNotify(void *pvParameters) {
    while (true) {
        if(notifyFlag){
          // digitalWrite(relay, HIGH)
          notify();
          delay(5000); // tạm dừng task 5s sau khi thông báo đến server
          predictFlag = true;
          notifyFlag = false;
          // digitalWrite(relay, LOW)
        }else {
          delay(500);
        }
    }
}

void loop() {
  //  unsigned long currentMillis = millis();
  // if (currentMillis - previousMillis >= timerInterval) {
  //   // predict1d();
  //   predict2d();
  //   previousMillis = currentMillis;
  // }
}
