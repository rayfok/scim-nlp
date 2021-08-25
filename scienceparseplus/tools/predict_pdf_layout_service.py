import tempfile

from fastapi import FastAPI, File, UploadFile
import requests
import aiofiles
import aiohttp

from predict_pdf_layout import build_layout_detection_model
from parse_pdf_with_cermine import run_and_parse_cermine

app = FastAPI()
predictor = build_layout_detection_model()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/detect/")
async def detect_upload_file(pdf_file: UploadFile = File(...)):

    with tempfile.TemporaryDirectory() as tempdir:
        async with aiofiles.open(f"{tempdir}/tmp.pdf", "wb") as out_file:
            content = await pdf_file.read()  # async read
            await out_file.write(content)  # async write

        layout = predictor.predict_pdf(f"{tempdir}/tmp.pdf")
    return {"layout": layout}

@app.get("/detect/")
async def detect_url(pdf_url: str):

    # Refer to https://stackoverflow.com/questions/35388332/how-to-download-images-with-aiohttp
    with tempfile.TemporaryDirectory() as tempdir:
        async with aiohttp.ClientSession() as session:
            async with session.get(pdf_url) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(f"{tempdir}/tmp.pdf", mode='wb')
                    await f.write(await resp.read())
                    await f.close()

        layout = predictor.predict_pdf(f"{tempdir}/tmp.pdf")

    return {"url": pdf_url, "layout": layout}


@app.post("/cermine/")
async def detect_upload_file_with_cermine_engine(pdf_file: UploadFile = File(...)):

    with tempfile.TemporaryDirectory() as tempdir:
        async with aiofiles.open(f"{tempdir}/tmp.pdf", "wb") as out_file:
            content = await pdf_file.read()  # async read
            await out_file.write(content)  # async write

        layout = run_and_parse_cermine(f"{tempdir}/tmp.pdf")
    if layout:
        return {"message": "Successfully parsed the PDF using the CERMINE engine.", "layout": layout}
    else:
        return {"message": "Fail to parse the PDF using the CERMINE engine. Possibly due to problematic PDF files.", "layout": []}

@app.get("/cermine/")
async def detect_url_with_cermine_engine(pdf_url: str):

    # Refer to https://stackoverflow.com/questions/35388332/how-to-download-images-with-aiohttp
    with tempfile.TemporaryDirectory() as tempdir:
        async with aiohttp.ClientSession() as session:
            async with session.get(pdf_url) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(f"{tempdir}/tmp.pdf", mode='wb')
                    await f.write(await resp.read())
                    await f.close()

        layout = run_and_parse_cermine(f"{tempdir}/tmp.pdf")

    if layout:
        return {"message": "Successfully parsed the PDF using the CERMINE engine.", "layout": layout}
    else:
        return {"message": "Fail to parse the PDF using the CERMINE engine. Possibly due to problematic PDF files.", "layout": []}
