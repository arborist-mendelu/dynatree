FROM python:3.12

WORKDIR /dynatree

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY scripts ./scripts/
RUN mkdir /root/solara_log

EXPOSE 8700
WORKDIR scripts
CMD ["solara", "run", "solara_app.py", "--host=0.0.0.0", "--port=8700"]
