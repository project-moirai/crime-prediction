# pylint: disable=no-name-in-module
# pylint: disable=no-self-argument

from fastapi import FastAPI, Response, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import uvicorn
from fastapi.responses import PlainTextResponse, StreamingResponse
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from models.xgboost.predict import XGBoostPredictionModel
from openai import AsyncAzureOpenAI


app = FastAPI(title="Project MoirAI")
openai_endpoint = "https://<CHANGEME>.cognitiveservices.azure.com/"

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

credentials = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credentials, "https://cognitiveservices.azure.com/.default"
)


class OpenAICompletion:
    def __init__(self, endpoint):
        self.deployment = "<CHANGEME>"
        api_version = "2024-12-01-preview"

        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
        )

    async def complete(self, system_prompt, user_prompt):
        try:
            # Enable streaming
            stream = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4096,
                temperature=1.0,
                top_p=1.0,
                model=self.deployment,
                stream=True,
            )

            async for event in stream:
                if event.choices and event.choices[0].delta.content is not None:
                    yield event.choices[0].delta.content

        except Exception as e:
            yield f"[Error: {str(e)}]"


@app.get("/")
async def main(
    request: Request,
):
    return PlainTextResponse(
        "OK",
        status_code=200,
    )


@app.get("/predict")
async def predict_tokens(
    request: Request,
    date: str,
    startTime: str,
    endTime: str,
    southWestLat: float,
    southWestLng: float,
    northEastLat: float,
    northEastLng: float,
):

    start = int(startTime.split(":")[0])
    end = int(endTime.split(":")[0])
    if end - start > 12:
        return PlainTextResponse(
            "Time range too big",
            status_code=400,
        )

    resolution_horizontal = 8
    resolution_vertical = 6
    steps_lat = (northEastLat - southWestLat) / resolution_vertical
    steps_lng = (northEastLng - southWestLng) / resolution_horizontal
    model = XGBoostPredictionModel()

    positions = []
    for i in range(1, resolution_vertical - 1):
        for j in range(1, resolution_horizontal - 1):
            lat = southWestLat + steps_lat * i
            lng = southWestLng + steps_lng * j
            positions.append((lat, lng))

    predictions = []
    for time_val in range(start, end + 1):
        for position in positions:
            lat = position[0]
            lng = position[1]
            category, score = model.predict(date, str(time_val) + ":00", lat, lng)
            if score < 0.6:
                continue
            if category != "no-incident":
                predictions.append(
                    {
                        "time": time_val,
                        "lat": lat,
                        "lng": lng,
                        "incident_probability": 100 * score,
                        "category": category,
                    }
                )

    completion = OpenAICompletion(openai_endpoint)
    system_prompt: str = "<CHANGEME>"
    user_prompt: str = str(predictions)

    async def token_stream():
        is_first = True
        async for token in completion.complete(system_prompt, user_prompt):
            if is_first:
                yield "<PREDICTIONS>" + json.dumps(predictions) + "</PREDICTIONS>"
                is_first = False
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
