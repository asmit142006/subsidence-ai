import joblib
import pandas as pd


# Load the trained AI
model = joblib.load("models/subsidence_model.pkl")
encoder = joblib.load("models/risk_encoder.pkl")


print("====================================")
print("   SUBSIDENCE AI SENSOR TEST")
print("====================================")
print()


# Enter sensor values
mpu_acc_x = float(input("MPU Acc X (g): "))
mpu_acc_y = float(input("MPU Acc Y (g): "))
mpu_acc_z = float(input("MPU Acc Z (g): "))

gyro_x = float(input("Gyro X (dps): "))
gyro_y = float(input("Gyro Y (dps): "))
gyro_z = float(input("Gyro Z (dps): "))

adxl_acc_x = float(input("ADXL Acc X (g): "))
adxl_acc_y = float(input("ADXL Acc Y (g): "))
adxl_acc_z = float(input("ADXL Acc Z (g): "))


# Put the readings into the same order used during training
input_data = pd.DataFrame([{
    "mpu_acc_x_g": mpu_acc_x,
    "mpu_acc_y_g": mpu_acc_y,
    "mpu_acc_z_g": mpu_acc_z,
    "gyro_x_dps": gyro_x,
    "gyro_y_dps": gyro_y,
    "gyro_z_dps": gyro_z,
    "adxl_acc_x_g": adxl_acc_x,
    "adxl_acc_y_g": adxl_acc_y,
    "adxl_acc_z_g": adxl_acc_z
}])


# Ask the AI for a prediction
prediction = model.predict(input_data)

# Convert the number back to the risk name
risk = encoder.inverse_transform(prediction)[0]


print()
print("====================================")
print("AI PREDICTION")
print("====================================")
print("Predicted risk:", risk.upper())
print("====================================")