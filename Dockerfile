FROM python:3.12-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p templates models/vit

COPY templates/index.html templates/
COPY models/inception.pth models/
COPY models/vit.pth models/
COPY models/vit/ models/vit/

COPY app.py vit.py inception.py ./

EXPOSE 5000

CMD ["python3", "app.py"] 