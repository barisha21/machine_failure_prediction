import numpy as np


def predict_machine_failure(

    model,
    scaler,
    encoder,
    machine_type,
    air_temp,
    process_temp,
    rotational_speed,
    torque,
    tool_wear

):

    machine_encoded = encoder.transform([machine_type])[0]

    input_data = np.array([[
        machine_encoded,
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    return prediction[0]