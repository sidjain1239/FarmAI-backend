from fastapi import FastAPI, Request
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = FastAPI()
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # Add this import


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/croprecommendation")
async def croprecommendation(request: Request):
    """
    # Postman Body:
    {
      "N": 90,
      "P": 42,
      "K": 43,
      "temperature": 20.87,
      "humidity": 82.0,
      "ph": 6.5,
      "rainfall": 202.93
    }
    """
    data = await request.json()
    input_data = pd.DataFrame({k: [data[k]] for k in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]})
    scaler = joblib.load('crop_r/crop_recommendation_scaler.pkl')
    encoder = joblib.load('crop_r/crop_recommendation_label_encoder.pkl')
    model = load_model('crop_r/crop_recommendation_model.keras')
    samples_scaled = scaler.transform(input_data)
    predictions = model.predict(samples_scaled)
    probs = predictions[0]
    top_3_indices = probs.argsort()[-3:][::-1]
    return {
        "predictions": [
            {
                "crop": encoder.inverse_transform([i])[0],
                "confidence": float(probs[i] * 100)
            }
            for i in top_3_indices
        ]
    }

@app.post("/cropitype")
async def cropwu(request: Request):
    """
    # Postman Body:
    {
      "Crop": "rice",
      "Season": "kharif",
      "Soil_Type": "clay",
      "Rainfall (mm)": 1200,
      "Temperature (°C)": 28
    }
    """
    data = await request.json()
    model = load_model('crop_wu/irrigation_type_ann_model.keras')
    label_encoder = joblib.load('crop_wu/irrigation_type_label_encoder.pkl')
    X_columns = joblib.load('crop_wu/irrigation_type_columns.pkl')
    sample = pd.DataFrame({k: [data[k]] for k in ["Crop", "Season", "Soil_Type", "Rainfall (mm)", "Temperature (°C)"]})
    sample_encoded = pd.get_dummies(sample).reindex(columns=X_columns, fill_value=0)
    pred_probs = model.predict(sample_encoded.astype('float32').to_numpy())
    pred_class_index = np.argmax(pred_probs, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([pred_class_index])[0]
    return {
        "predicted_irrigation_type": predicted_label,
        "confidence": float(pred_probs[0][pred_class_index] * 100)
    }

@app.post("/waterusage")
async def waterusage(request: Request):
    """
    # Postman Body:
    {
      "Soil_Type": "loamy",
      "Rainfall (mm)": 950,
      "Temperature (°C)": 27,
      "Area(ha)": 700,
      "Crop": "rice",
      "Season": "kharif",
      "Irrigation_Type": "surface"
    }
    """
    data = await request.json()
    model = load_model('crop_wu/water_usage_ann_model.keras')
    encoders = joblib.load('crop_wu/water_usage_encoders.pkl')
    X_columns = joblib.load('crop_wu/water_usage_columns.pkl')
    sample = pd.DataFrame([data])
    for col, encoder in encoders.items():
        sample[col] = encoder.transform(sample[col])
    sample = sample[X_columns]
    prediction = model.predict(sample)
    return {"predicted_water_usage": float(prediction[0][0])}
@app.post("/fertilizer")
async def fertilizer(request: Request):
    """
    # Postman Body:
    {
      "Temperature": 26,
      "Humidity": 78,
      "Moisture": 28,
      "Nitrogen": 80,
      "Phosphorous": 90,
      "Potassium": 30,
      "Soil": "Loamy Soil",
      "Crop": "rice",
      "Rainfall": 33,
      "PH": 33,
      "Carbon": 33
    }
    """
    data = await request.json()
    model = joblib.load('fer_r/fertilizer_model_label_encoded.pkl')
    feature_encoders = joblib.load('fer_r/fertilizer_feature_encoders.pkl')
    fertilizer_encoder = joblib.load('fer_r/fertilizer_target_encoder.pkl')
    feature_columns = joblib.load('fer_r/fertilizer_feature_columns.pkl')
    fertilizer_remark_mapping = joblib.load('fer_r/fertilizer_remark_mapping.pkl')
    sample = pd.DataFrame([data])
    for col, encoder in feature_encoders.items():
        sample[col] = encoder.transform(sample[col])
    sample = sample[feature_columns]
    predicted_label_encoded = model.predict(sample)[0]
    predicted_fertilizer = fertilizer_encoder.inverse_transform([predicted_label_encoded])[0]
    remark = fertilizer_remark_mapping.get(predicted_fertilizer, "No remark available")
    return {
        "predicted_fertilizer": predicted_fertilizer,
        "remark": remark
    }

@app.post("/pfq")
async def pfq(request: Request):
    """
    # Postman Body:
    {
      "Crop_Type": "rice",
      "Season": "rabi",
      "Soil_Type": "loamy",
      "Temperature": 29,
      "Humidity": 78,
      "Rainfall": 120,
      "PH": 6.5,
      "Yield(Tons)": 5,
      "Irrigation_Type": "drip"
    }
    """
    data = await request.json()
    fertilizer_model = load_model('pf_q/fertilizer_model_ann.keras')
    pesticide_model = load_model('pf_q/pesticide_model_ann.keras')
    encoders = joblib.load('pf_q/pfq_feature_encoders.pkl')
    feature_columns = joblib.load('pf_q/pfq_feature_columns.pkl')
    sample = pd.DataFrame([data])
    for col, le in encoders.items():
        sample[col] = le.transform(sample[col])
    for col in feature_columns:
        if col not in sample.columns:
            sample[col] = 0
    sample = sample[feature_columns]
    input_array = sample.to_numpy()
    fertilizer_pred = fertilizer_model.predict(input_array, verbose=0)[0][0]
    pesticide_pred = pesticide_model.predict(input_array, verbose=0)[0][0]
    return {
        "predicted_fertilizer_usage_tons": float(fertilizer_pred),
        "predicted_pesticide_usage_kg": float(pesticide_pred)
    }

@app.post("/cropyield")
async def crop_yield(request: Request):
    """
    # Postman Body:
    {
      "temperature": 30.5,
      "rainfall": 800,
      "humidity": 85,
      "soil_ph": 18.0,
      "area": 200
    }
    """
    data = await request.json()
    # Prepare input as 2D array for scaler/model
    sample = np.array([[data["temperature"], data["rainfall"], data["humidity"], data["soil_ph"], data["area"]]])
    scaler = joblib.load("crop_yp/scaler_crop_yield.pkl")
    model = load_model("crop_yp/crop_yield_predictor.keras")
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    return {"predicted_crop_yield": float(prediction[0][0])}

@app.post("/agri-analytics")
async def agri_analytics(request: Request):
    """
    # Postman Body - Input only the initial parameters:
    {
      "N": 90,
      "P": 42,
      "K": 43,
      "temperature": 28,
      "rainfall": 800,
      "humidity": 85,
      "ph": 6.5,
      "Soil_Type": "loamy",
      "Season": "kharif",
      "Area(ha)": 200,
      "Moisture": 28,
      "Carbon": 33
    }
    """
    data = await request.json()

    # Normalize input data
    normalized_data = {
        "N": data.get("N"),
        "P": data.get("P"),
        "K": data.get("K"),
        "temperature": data.get("temperature"),
        "rainfall": data.get("rainfall"),
        "humidity": data.get("humidity"),
        "ph": data.get("ph"),
        "Soil_Type": data.get("Soil_Type"),
        "Season": data.get("Season"),
        "Area(ha)": data.get("Area(ha)"),
        "Moisture": data.get("Moisture", 28),  # Default value if missing
        "Carbon": data.get("Carbon", 33)       # Default value if missing
    }

    results = {}

    try:
        # Step 1: Crop Recommendation
        crop_input = pd.DataFrame([{
            "N": normalized_data["N"],
            "P": normalized_data["P"],
            "K": normalized_data["K"],
            "temperature": normalized_data["temperature"],
            "humidity": normalized_data["humidity"],
            "ph": normalized_data["ph"],
            "rainfall": normalized_data["rainfall"]
        }])
        scaler = joblib.load('crop_r/crop_recommendation_scaler.pkl')
        encoder = joblib.load('crop_r/crop_recommendation_label_encoder.pkl')
        crop_model = load_model('crop_r/crop_recommendation_model.keras')
        crop_scaled = scaler.transform(crop_input)
        crop_predictions = crop_model.predict(crop_scaled)
        crop_probs = crop_predictions[0]
        top_crop_index = crop_probs.argmax()
        predicted_crop = encoder.inverse_transform([top_crop_index])[0]
        results["crop_recommendation"] = {
            "crop": predicted_crop,
            "confidence": float(crop_probs[top_crop_index] * 100)
        }

        # Step 2: Irrigation Type
        irrigation_input = pd.DataFrame([{
            "Crop": predicted_crop,
            "Season": normalized_data["Season"],
            "Soil_Type": normalized_data["Soil_Type"],
            "Rainfall (mm)": normalized_data["rainfall"],
            "Temperature (°C)": normalized_data["temperature"]
        }])
        irrigation_model = load_model('crop_wu/irrigation_type_ann_model.keras')
        irrigation_label_encoder = joblib.load('crop_wu/irrigation_type_label_encoder.pkl')
        irrigation_columns = joblib.load('crop_wu/irrigation_type_columns.pkl')
        irrigation_encoded = pd.get_dummies(irrigation_input).reindex(columns=irrigation_columns, fill_value=0)
        irrigation_probs = irrigation_model.predict(irrigation_encoded.astype('float32').to_numpy())
        irrigation_class_index = np.argmax(irrigation_probs, axis=1)[0]
        predicted_irrigation_type = irrigation_label_encoder.inverse_transform([irrigation_class_index])[0]
        results["irrigation_type"] = {
            "predicted_irrigation_type": predicted_irrigation_type,
            "confidence": float(irrigation_probs[0][irrigation_class_index] * 100)
        }

        # Step 3: Water Usage
        water_input = pd.DataFrame([{
            "Soil_Type": normalized_data["Soil_Type"],
            "Rainfall (mm)": normalized_data["rainfall"],
            "Temperature (°C)": normalized_data["temperature"],
            "Area(ha)": normalized_data["Area(ha)"],
            "Crop": predicted_crop,
            "Season": normalized_data["Season"],
            "Irrigation_Type": predicted_irrigation_type
        }])
        water_model = load_model('crop_wu/water_usage_ann_model.keras')
        water_encoders = joblib.load('crop_wu/water_usage_encoders.pkl')
        water_columns = joblib.load('crop_wu/water_usage_columns.pkl')
        for col, encoder in water_encoders.items():
            water_input[col] = encoder.transform(water_input[col])
        water_input = water_input[water_columns]
        water_prediction = water_model.predict(water_input)
        results["water_usage"] = {"predicted_water_usage": float(water_prediction[0][0])}

        # Step 4: Fertilizer Recommendation
        fert_input = pd.DataFrame([{
            "Temperature": normalized_data["temperature"],
            "Humidity": normalized_data["humidity"],
            "Nitrogen": normalized_data["N"],
            "Phosphorous": normalized_data["P"],
            "Potassium": normalized_data["K"],
            "Soil": normalized_data["Soil_Type"],
            "Crop": predicted_crop,
            "Rainfall": normalized_data["rainfall"],
            "PH": normalized_data["ph"],
            "Moisture": normalized_data["Moisture"],
            "Carbon": normalized_data["Carbon"]
        }])
        fert_model = joblib.load('fer_r/fertilizer_model_label_encoded.pkl')
        fert_encoders = joblib.load('fer_r/fertilizer_feature_encoders.pkl')
        fert_label_encoder = joblib.load('fer_r/fertilizer_target_encoder.pkl')
        fert_columns = joblib.load('fer_r/fertilizer_feature_columns.pkl')
        fert_remark_mapping = joblib.load('fer_r/fertilizer_remark_mapping.pkl')
        for col, encoder in fert_encoders.items():
            fert_input[col] = encoder.transform(fert_input[col])
        fert_input = fert_input[fert_columns]
        fert_prediction = fert_model.predict(fert_input)[0]
        fert_label = fert_label_encoder.inverse_transform([fert_prediction])[0]
        fert_remark = fert_remark_mapping.get(fert_label, "No remark available")
        results["fertilizer_recommendation"] = {
            "predicted_fertilizer": fert_label,
            "remark": fert_remark
        }

        # Step 5: Crop Yield
        yield_input = np.array([[
            normalized_data["temperature"],
            normalized_data["rainfall"],
            normalized_data["humidity"],
            normalized_data["ph"],
            normalized_data["Area(ha)"]
        ]])
        yield_scaler = joblib.load("crop_yp/scaler_crop_yield.pkl")
        yield_model = load_model("crop_yp/crop_yield_predictor.keras")
        yield_scaled = yield_scaler.transform(yield_input)
        yield_prediction = yield_model.predict(yield_scaled)
        results["crop_yield"] = {"predicted_crop_yield": float(yield_prediction[0][0])}
        # Step 6: Fertilizer and Pesticide Quantity
        try:
            pfq_input = pd.DataFrame([{
                "Crop_Type": predicted_crop,
                "Season": normalized_data["Season"],
                "Soil_Type": normalized_data["Soil_Type"],
                "Temperature": normalized_data["temperature"],
                "Humidity": normalized_data["humidity"],
                "Rainfall": normalized_data["rainfall"],
                "PH": normalized_data["ph"],
                "Yield(Tons)": results["crop_yield"]["predicted_crop_yield"],  # Use predicted crop yield
                "Irrigation_Type": predicted_irrigation_type or "drip"  # Default to "drip" if irrigation type is missing
            }])
            pfq_fertilizer_model = load_model('pf_q/fertilizer_model_ann.keras')
            pfq_pesticide_model = load_model('pf_q/pesticide_model_ann.keras')
            pfq_encoders = joblib.load('pf_q/pfq_feature_encoders.pkl')
            pfq_feature_columns = joblib.load('pf_q/pfq_feature_columns.pkl')

            # Encode input data
            for col, encoder in pfq_encoders.items():
                try:
                    pfq_input[col] = encoder.transform(pfq_input[col])
                except ValueError as e:
                    # Handle unseen labels by defaulting to the first class
                    if "contains previously unseen labels" in str(e):
                        pfq_input[col] = encoder.transform([encoder.classes_[0]])[0]
            for col in pfq_feature_columns:
                if col not in pfq_input.columns:
                    pfq_input[col] = 0
            pfq_input = pfq_input[pfq_feature_columns]
            pfq_array = pfq_input.to_numpy()

            # Predict fertilizer and pesticide quantities
            fertilizer_quantity = pfq_fertilizer_model.predict(pfq_array, verbose=0)[0][0]
            pesticide_quantity = pfq_pesticide_model.predict(pfq_array, verbose=0)[0][0]

            results["fertilizer_pesticide_quantity"] = {
                "predicted_fertilizer_usage_tons": float(fertilizer_quantity),
                "predicted_pesticide_usage_kg": float(pesticide_quantity)
            }
        except Exception as e:
            # Handle errors and retry with "drip" as the default irrigation type
            try:
                pfq_input["Irrigation_Type"] = "drip"  # Default to "drip"
                for col, encoder in pfq_encoders.items():
                    try:
                        pfq_input[col] = encoder.transform(pfq_input[col])
                    except ValueError as e:
                        # Handle unseen labels by defaulting to the first class
                        if "contains previously unseen labels" in str(e):
                            pfq_input[col] = encoder.transform([encoder.classes_[0]])[0]
                pfq_input = pfq_input[pfq_feature_columns]
                pfq_array = pfq_input.to_numpy()

                # Retry prediction with "drip"
                fertilizer_quantity = pfq_fertilizer_model.predict(pfq_array, verbose=0)[0][0]
                pesticide_quantity = pfq_pesticide_model.predict(pfq_array, verbose=0)[0][0]

                results["fertilizer_pesticide_quantity"] = {
                    "predicted_fertilizer_usage_tons": float(fertilizer_quantity),
                    "predicted_pesticide_usage_kg": float(pesticide_quantity),
                    "note": "Defaulted to 'drip' irrigation type due to error"
                }
            except Exception as retry_error:
                results["fertilizer_pesticide_quantity"] = {
                    "error": f"Failed to predict fertilizer and pesticide quantities: {str(retry_error)}"
                }

          
          
    except Exception as e:
        results["error"] = str(e)

    return results
@app.get("/debug/categories")
async def get_categories():
    """Get all valid categories from the encoders"""
    encoders = joblib.load('fer_r/fertilizer_feature_encoders.pkl')
    return {
        "soil_types": encoders["Soil"].classes_.tolist(),
        "crop_types": encoders["Crop"].classes_.tolist()
    }