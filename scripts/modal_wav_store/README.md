Lynkup Modal WAV Store
======================

This Modal app exposes a web endpoint that accepts an `audioUrl`, validates a header key, and stores the WAV on a persistent Modal Volume. It mirrors the LinkProServerApi contract by expecting the `X-CUSTOM-API-KEY` header and returning `{ "accepted": true }` immediately while work runs in the background.

References: see Modal docs for quickstarts and volumes: [Guide](https://modal.com/docs/guide), [Volumes](https://modal.com/docs/guide/volumes). The overall goal is aligned with Modal's batch transcription guidance: [Transcribe 100x faster/cheaper](https://modal.com/blog/fast-cheap-batch-transcription).

Setup
-----

1. Install dependencies (in your Python environment):

   ```bash
   pip install -r ../../requirements.txt
   ```

2. Authenticate to Modal (one-time):

   ```bash
   modal setup
   ```

3. Configure the header key expected by the endpoint (matches your Azure Function client env):

   ```bash
   # Set as an env var in your Modal environment
   export VIDEO_API_SERVER_HEADER_KEY=your-secret-key
   ```

   You can also inject this via Modal Secrets or the `env` parameter on `@app.function` if preferred.

Deploy & Run
------------

You can run locally (hot-reload) or deploy.

- Local dev server (prints a temporary URL):

  ```bash
  modal serve app.py
  ```

- Deploy to a stable URL:

  ```bash
  modal deploy app.py
  ```

Endpoints
---------

- `POST /transcribeAndAnalyze`

  Headers:

  - `X-CUSTOM-API-KEY: <your-secret-key>`

  JSON body:

  ```json
  { "audioUrl": "https://example.com/path/to/file.wav", "sessionId": "optional-session" }
  ```

  Response:

  ```json
  { "accepted": true }
  ```

  The download to a Modal Volume named `lynkup-audio-volume` is executed in the background.

- `GET /healthz`

  Simple health check.

Integrating with LynkupVideoTranscription
-----------------------------------------

Set these environment variables for the Azure Function app so it calls the Modal endpoint exactly like the current server:

- `VIDEO_API_SERVER_URL`: set to the Modal base URL returned by `modal deploy` or `modal serve` (e.g., `https://<your-app>.modal.run`).
- `VIDEO_API_SERVER_AUDIO_PATH`: set to `/transcribeAndAnalyze`.
- `VIDEO_API_SERVER_HEADER_KEY`: same value configured for the Modal service.

Your existing `VideoService.sendAudioLocation` will then POST to `VIDEO_API_SERVER_URL + VIDEO_API_SERVER_AUDIO_PATH` with the header and `{ audioUrl }` body, matching the current LinkProServerApi contract.
