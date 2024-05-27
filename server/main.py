from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from typing import List, Dict, Optional

from datetime import datetime
from FatrCode import FatrCode

import json



app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/sensorVals")
async def get_senorVals():
    sensor_values = FatrCode.as_dict()
    
    response_data = {
        "num_sensors": len(sensor_values),
        "available_sensors": sensor_values
    }
    
    json_response = json.dumps(response_data, ensure_ascii=False)
    
    return JSONResponse(content=json.loads(json_response), media_type="application/json; charset=utf-8")

class SensorValue(BaseModel):
    acquired_time: datetime
    sensor_values: Optional[List[Dict[str, float]]]

@app.post("/isAbnormal")
async def post_isAbnormal(sensor_values: SensorValue):
    
    return {
        "message": "This is dummy.",
        "input_sensor_values": sensor_values,
        "isAbnormal": [0, 0, 1]
    }

